#!/usr/bin/env bash
# Build causal-conv1d (latest sdist on Aliyun) in the sr_opsd env.
# Reuses the nvcc 12.8 toolchain from the vision-opd conda env (torch is cu128).
set -euo pipefail

VOPD=/home/c30084464/miniconda3/envs/vision-opd
export CUDA_HOME="$VOPD"
export CUDACXX="$VOPD/bin/nvcc"
export CPATH="$VOPD/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$VOPD/targets/x86_64-linux/lib:$VOPD/targets/x86_64-linux/lib/stubs:$VOPD/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$VOPD/targets/x86_64-linux/lib:$VOPD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# H20 = Hopper = sm_90.
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=32
# Force a local source build instead of downloading a prebuilt wheel from github
# (the github release wheel guess also mismatches our torch version).
export CAUSAL_CONV1D_FORCE_BUILD=TRUE

echo "== nvcc used =="
nvcc --version | tail -3
echo "== torch =="
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)"
echo "== start build $(date) =="

python -m pip install -v --no-build-isolation --no-deps \
  -i https://mirrors.aliyun.com/pypi/simple/ \
  "causal-conv1d==1.6.2.post1"

echo "== build done $(date) =="
python -c "import causal_conv1d; print('causal_conv1d', causal_conv1d.__version__, causal_conv1d.__file__)"
