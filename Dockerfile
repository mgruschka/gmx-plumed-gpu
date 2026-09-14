# Portable GPU image: GROMACS + PLUMED + Python (OpenMM / MDTraj / MDAnalysis)
#
# Runs under Apptainer (--nv) on Midway and Docker (--gpus all) on AWS.
# The image carries the CUDA *toolkit*; the NVIDIA *driver* is injected by the host,
# which is what makes one image portable across both resources.
#
# ---- knobs to check before building ----
#   CUDA version : must be <= the driver on your GPU host. Check `nvidia-smi` on a
#                  Midway3 GPU node (top-right "CUDA Version"). Lower the base tag to
#                  match if needed (e.g. 11.8.0-devel-ubuntu22.04) and drop sm_90 below.
#   GPU targets  : GMX_CUDA_TARGET_SM lists the architectures compiled in.
#                  70=V100, 75=T4, 80=A100, 86=A10/RTX, 89=L4, 90=H100.
#                  Trim to what you'll actually run on to cut build time / image size.

FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget bzip2 ca-certificates \
    libfftw3-dev libgsl-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------- PLUMED ----------
ARG PLUMED_VERSION=2.9.0
RUN wget -q https://github.com/plumed/plumed2/releases/download/v${PLUMED_VERSION}/plumed-src-${PLUMED_VERSION}.tgz \
 && tar xf plumed-src-${PLUMED_VERSION}.tgz && cd plumed-${PLUMED_VERSION} \
 && ./configure --prefix=/usr/local \
 && make -j"$(nproc)" && make install && ldconfig \
 && cd .. && rm -rf plumed-${PLUMED_VERSION}*
ENV PLUMED_KERNEL=/usr/local/lib/libplumedKernel.so

# ---------- GROMACS (CUDA, thread-MPI), patched with PLUMED ----------
# thread-MPI (GMX_MPI=OFF) -> no host-MPI dependency, single-node + multi-GPU.
# For multi-walker metadynamics use PLUMED file-based WALKERS (no MPI needed).
ARG GROMACS_VERSION=2023.2
RUN wget -q https://ftp.gromacs.org/gromacs/gromacs-${GROMACS_VERSION}.tar.gz \
 && tar xf gromacs-${GROMACS_VERSION}.tar.gz && cd gromacs-${GROMACS_VERSION} \
 && plumed patch -p -e gromacs-${GROMACS_VERSION} \
 && mkdir build && cd build \
 && cmake .. \
      -DGMX_GPU=CUDA \
      -DGMX_MPI=OFF \
      -DGMX_SIMD=AVX2_256 \
      -DGMX_BUILD_OWN_FFTW=OFF \
      -DGMX_CUDA_TARGET_SM="70;75;80;86;89;90" \
      -DCMAKE_INSTALL_PREFIX=/usr/local/gromacs \
 && make -j"$(nproc)" && make install \
 && cd ../.. && rm -rf gromacs-${GROMACS_VERSION}*
ENV PATH=/usr/local/gromacs/bin:$PATH

# ---------- Python / OpenMM via micromamba ----------
# conda-forge is the reliable path for GPU OpenMM. The PLUMED python wrapper is
# pip-installed so it binds to the source-built kernel above (PLUMED_KERNEL).
ENV MAMBA_ROOT_PREFIX=/opt/conda
RUN wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xvj -C /usr/local bin/micromamba \
 && micromamba create -y -p /opt/conda -c conda-forge \
      python=3.11 openmm mdtraj mdanalysis \
      numpy scipy pandas matplotlib \
 && /opt/conda/bin/pip install --no-cache-dir plumed \
 && micromamba clean -a -y
ENV PATH=/opt/conda/bin:$PATH

# convenience for interactive shells (GMXLIB, completion, etc.)
RUN echo "source /usr/local/gromacs/bin/GMXRC" >> /etc/bash.bashrc

WORKDIR /work
