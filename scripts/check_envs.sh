#!/usr/bin/env bash
set -uo pipefail
PKGS="spatialstack lmms_eval torch transformers deepspeed modelscope accelerate flash_attn vggt qwen_vl"
for e in base slime spatialstack-qwen35 sr_opsd vision-opd; do
  echo "==================== $e ===================="
  if [ "$e" = "base" ]; then p="/home/c30084464/miniconda3"; else p="/home/c30084464/miniconda3/envs/$e"; fi
  du -sh "$p" 2>/dev/null | awk '{print "size:",$1}'
  "$p/bin/python" -c "import sys;print('python',sys.version.split()[0])" 2>/dev/null
  for pkg in $PKGS; do
    v=$("$p/bin/python" -c "import importlib; m=importlib.import_module('$pkg'); print(getattr(m,'__version__','?'))" 2>/dev/null)
    [ -n "$v" ] && echo "  $pkg $v"
  done
  echo ""
done