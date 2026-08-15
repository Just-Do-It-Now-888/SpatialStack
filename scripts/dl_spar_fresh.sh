#!/usr/bin/env bash
# Fresh re-download (overwrite) of 4 suspect SPAR chunks via curl single-stream (no ranges -> no 403).
# Retry until correct size. Parallel. setsid-survivable.
set -uo pipefail
cd /home/c30084464/Documents/code/SpatialStack_OPSD
REV=976c19177468eabe64e9e2dd0f0450cd32dacc1f
BASE="https://hf-mirror.com/datasets/jasonzhango/SPAR-7M/resolve/$REV"
EXPECTED=10737418240
OUT=./data/media/spar
LOG=./output/spar_chunks_dl.log
mkdir -p "$OUT" ./output
exec >> "$LOG" 2>&1
echo "=== spar FRESH redl start $(date) ==="

dl_one() {
  local c=$1
  local f="$OUT/spar-$c.tar.gz"
  for attempt in $(seq 1 30); do
    sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$sz" -ge "$EXPECTED" ]; then
      echo "[fresh-dl] spar-$c already complete ($sz), skip"
      return 0
    fi
    echo "[fresh-dl] spar-$c attempt $attempt at $(date)"
    curl -sL --connect-timeout 30 --max-time 3000 -o "$f" "$BASE/spar-$c.tar.gz" \
      || echo "[fresh-dl] spar-$c curl rc=$? (retry)"
  done
  sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
  [ "$sz" -ge "$EXPECTED" ] && { echo "[fresh-dl] spar-$c complete ($sz)"; return 0; }
  echo "[fresh-dl] spar-$c FAILED: $sz < $EXPECTED"
  return 1
}
export -f dl_one
export BASE EXPECTED OUT

for c in 03 05 06 11; do
  dl_one "$c" &
done
wait
echo "=== spar FRESH redl end $(date) ==="
for c in 00 01 02 03 04 05 06 07 08 09 10 11 12 13; do
  echo "spar-$c: $(stat -c %s $OUT/spar-$c.tar.gz 2>/dev/null || echo MISSING)"
done
