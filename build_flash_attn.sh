#!/usr/bin/env bash
# Build flash_attn 2.8.3 (SpatialStack-pinned) in the sr_opsd env.
# Reuses the nvcc 12.8 toolchain from the vision-opd conda env (torch is cu128).
set -euo pipefail

VOPD=/home/c30084464/miniconda3/envs/vision-opd
export CUDA_HOME="$VOPD"
export CUDACXX="$VOPD/bin/nvcc"
# NOTE: do NOT prepend $VOPD/bin to PATH, otherwise `python`/`pip` resolve to the
# vision-opd env and flash_attn gets installed into vision-opd instead of sr_opsd.
# PyTorch's cpp_extension finds nvcc via CUDA_HOME/bin/nvcc.
export CPATH="$VOPD/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$VOPD/targets/x86_64-linux/lib:$VOPD/targets/x86_64-linux/lib/stubs:$VOPD/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$VOPD/targets/x86_64-linux/lib:$VOPD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# H20 = Hopper = sm_90. Only build sm_90 to keep compile fast.
export TORCH_CUDA_ARCH_LIST="9.0"
export FLASH_ATTN_CUDA_ARCHS="90"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=32

echo "== nvcc used =="
nvcc --version | tail -3
echo "== torch =="
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)"
echo "== start build $(date) =="

python -m pip install -v --no-build-isolation --no-deps --no-binary flash-attn \
  -i https://mirrors.aliyun.com/pypi/simple/ \
  "flash_attn==2.8.3"

echo "== build done $(date) =="
python -c "import flash_attn, flash_attn_2_cuda; print('flash_attn', flash_attn.__version__, flash_attn.__file__)"
