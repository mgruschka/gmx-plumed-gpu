# Portable GPU image: GROMACS + PLUMED + Python (PyTorch / MDTraj / MDAnalysis / RDKit / OpenBabel)
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

# Parallel build jobs. Default is conservative for a local WSL build with limited RAM;
# pass --build-arg JOBS=$(nproc) on a beefier host.
ARG JOBS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget bzip2 ca-certificates \
    libfftw3-dev libgsl-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------- PLUMED ----------
ARG PLUMED_VERSION=2.9.4
RUN wget -q https://github.com/plumed/plumed2/releases/download/v${PLUMED_VERSION}/plumed-src-${PLUMED_VERSION}.tgz \
 && tar xf plumed-src-${PLUMED_VERSION}.tgz && cd plumed-${PLUMED_VERSION} \
 && ./configure --prefix=/usr/local \
 && make -j"${JOBS}" && make install && ldconfig \
 && cd .. && rm -rf plumed-${PLUMED_VERSION}*
ENV PLUMED_KERNEL=/usr/local/lib/libplumedKernel.so

# ---------- GROMACS (CUDA, thread-MPI), patched with PLUMED ----------
# GROMACS and PLUMED versions must be a matched pair: PLUMED only ships patches
# for specific GROMACS releases (see plumed2/patches/ for the chosen PLUMED tag).
# PLUMED 2.9.x -> gromacs-2023.5 or gromacs-2024.3.
# thread-MPI (GMX_MPI=OFF) -> no host-MPI dependency, single-node + multi-GPU.
# For multi-walker metadynamics use PLUMED file-based WALKERS (no MPI needed).
ARG GROMACS_VERSION=2023.5
# Trimmed to a single arch (A100) for local troubleshooting. Restore the full list
# for the cluster build: --build-arg GMX_CUDA_TARGET_SM="70;75;80;86;89;90"
ARG GMX_CUDA_TARGET_SM=80
RUN wget -q https://ftp.gromacs.org/gromacs/gromacs-${GROMACS_VERSION}.tar.gz \
 && tar xf gromacs-${GROMACS_VERSION}.tar.gz && cd gromacs-${GROMACS_VERSION} \
 && plumed patch -p -e gromacs-${GROMACS_VERSION} \
 && mkdir build && cd build \
 && cmake .. \
      -DGMX_GPU=CUDA \
      -DGMX_MPI=OFF \
      -DGMX_SIMD=AVX2_256 \
      -DGMX_BUILD_OWN_FFTW=OFF \
      -DGMX_CUDA_TARGET_SM="${GMX_CUDA_TARGET_SM}" \
      -DCMAKE_INSTALL_PREFIX=/usr/local/gromacs \
 && make -j"${JOBS}" && make install \
 && cd ../.. && rm -rf gromacs-${GROMACS_VERSION}*
ENV PATH=/usr/local/gromacs/bin:$PATH

# ---------- Python stack via micromamba ----------
# Self-contained env at /opt/conda/envs/toid, conda-forge only (no pip/conda mixing),
# with the exceptions listed at the pip step below.
#   pytorch=*=cuda12*  : conda-forge's CUDA build. The legacy `-c pytorch pytorch-cuda`
#                        recipe no longer works (PyTorch stopped publishing conda packages
#                        after 2.5) and silently resolves to conda-forge's CPU-only build.
#   cuda-version       : pins the CUDA runtime conda ships for torch/xgboost. Must be <=
#                        the host driver's CUDA version (`nvidia-smi`). Newer values pull
#                        newer torch; 12.2 matches the base image and is conservative.
#   CONDA_OVERRIDE_CUDA: there is no GPU/driver at build time, so tell the solver which
#                        __cuda virtual package to assume; otherwise CUDA builds are
#                        "not installable".
#   openbabel          : provides the obabel CLI.
ARG TORCH_CUDA_VERSION=12.2
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV TOID_ENV=/opt/conda/envs/toid
RUN wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xj -C /usr/local bin/micromamba \
 && CONDA_OVERRIDE_CUDA=${TORCH_CUDA_VERSION} \
    micromamba create -y -p ${TOID_ENV} --override-channels -c conda-forge \
      python=3.11 \
      numpy scipy pandas matplotlib seaborn scikit-learn h5py \
      mdtraj mdanalysis biopython \
      rdkit xgboost networkx tqdm pillow \
      "pytorch=*=cuda12*" "cuda-version=${TORCH_CUDA_VERSION}" gpytorch \
      openbabel \
      ipython jupyterlab ipykernel \
 && micromamba clean -a -y

# pip-only packages:
#   plumed         : the python wrapper is installed via pip (not conda) so it binds to the
#                    source-built kernel above via $PLUMED_KERNEL instead of pulling a second
#                    PLUMED kernel from conda-forge. Thin Cython shim; keep matched to PLUMED_VERSION.
#   PeptideBuilder : not packaged on conda-forge; pure Python on top of Biopython (already in env).
RUN ${TOID_ENV}/bin/pip install --no-cache-dir "plumed==${PLUMED_VERSION}" PeptideBuilder

# ---------- Runtime environment ----------
# Apptainer imports these into the container environment, so run_sim.sh / mini_sim.sh
# and the sbatch templates work without `module load` / `source activate` lines.
# thread-MPI build: no MPI runtime, the binary is `gmx` (not gmx_mpi), no mpirun.
# GMXLIB is deliberately left unset: forcefields/ lives in the bind-mounted repo, so
# pass it at run time, e.g. --env GMXLIB=/scratch/midway3/<user>/toid_explore/forcefields/
ENV PATH=${TOID_ENV}/bin:/usr/local/gromacs/bin:/usr/local/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/gromacs/lib:/usr/local/lib:$LD_LIBRARY_PATH
ENV OMP_NUM_THREADS=1

# convenience for interactive shells (GMXLIB, completion, etc.)
RUN echo "source /usr/local/gromacs/bin/GMXRC" >> /etc/bash.bashrc

WORKDIR /work
