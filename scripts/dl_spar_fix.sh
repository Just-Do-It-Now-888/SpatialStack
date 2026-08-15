#!/usr/bin/env bash
# Truncate the 4 suspect chunks to their curl-clean size, then curl -C - (clean single-stream) the rest.
set -uo pipefail
cd /home/c30084464/Documents/code/SpatialStack_OPSD
REV=976c19177468eabe64e9e2dd0f0450cd32dacc1f
BASE="https://hf-mirror.com/datasets/jasonzhango/SPAR-7M/resolve/$REV"
EXPECTED=10737418240
OUT=./data/media/spar
LOG=./output/spar_chunks_dl.log
exec >> "$LOG" 2>&1
echo "=== spar FIX-aria-gaps start $(date) ==="

# curl-clean sizes (before aria2c touched them); spar-11 was fully aria2c -> 0
declare -A CLEAN=( [03]=7948226560 [05]=6312914944 [06]=6235333149 [11]=0 )
rm -f "$OUT"/*.aria2

dl_one() {
  local c=$1 clean=${CLEAN[$c]}
  local f="$OUT/spar-$c.tar.gz"
  truncate -s "$clean" "$f" 2>/dev/null || { rm -f "$f"; }
  echo "[fix] spar-$c truncated to $clean"
  for attempt in $(seq 1 300); do
    sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$sz" -ge "$EXPECTED" ]; then echo "[fix] spar-$c complete ($sz)"; return 0; fi
    echo "[fix] spar-$c attempt $attempt (cur=$sz) at $(date)"
    curl -sL -C - --connect-timeout 30 --retry 20 --retry-delay 3 -o "$f" "$BASE/spar-$c.tar.gz" \
      || echo "[fix] spar-$c curl rc=$? (resume-retry)"
  done
  sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
  [ "$sz" -ge "$EXPECTED" ] && { echo "[fix] spar-$c complete ($sz)"; return 0; }
  echo "[fix] spar-$c FAILED: $sz"; return 1
}
export -f dl_one; export BASE EXPECTED OUT
declare -A CLEAN_EXPORT=("${CLEAN[@]}")
# pass CLEAN via env file
for k in "${!CLEAN[@]}"; do echo "$k ${CLEAN[$k]}"; done > /tmp/spar_clean_map.txt
export CLEAN_MAP=/tmp/spar_clean_map.txt

dl_one2() {
  local c=$1
  local clean=$(awk -v k=$c '$1==k{print $2}' $CLEAN_MAP)
  local f="$OUT/spar-$c.tar.gz"
  truncate -s "$clean" "$f" 2>/dev/null || { rm -f "$f"; }
  echo "[fix] spar-$c truncated to $clean"
  for attempt in $(seq 1 300); do
    sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$sz" -ge "$EXPECTED" ]; then echo "[fix] spar-$c complete ($sz)"; return 0; fi
    echo "[fix] spar-$c attempt $attempt (cur=$sz) at $(date)"
    curl -sL -C - --connect-timeout 30 --retry 20 --retry-delay 3 -o "$f" "$BASE/spar-$c.tar.gz" \
      || echo "[fix] spar-$c curl rc=$? (resume-retry)"
  done
  sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
  [ "$sz" -ge "$EXPECTED" ] && { echo "[fix] spar-$c complete ($sz)"; return 0; }
  echo "[fix] spar-$c FAILED: $sz"; return 1
}
export -f dl_one2
for c in 03 05 06 11; do dl_one2 "$c" & done
wait
echo "=== spar FIX-aria-gaps end $(date) ==="
for c in 03 05 06 11; do echo "spar-$c: $(stat -c %s $OUT/spar-$c.tar.gz 2>/dev/null)"; done
