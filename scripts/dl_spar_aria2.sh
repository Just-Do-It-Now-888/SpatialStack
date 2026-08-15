#!/usr/bin/env bash
# aria2c resume (infinite retries) for 3 remaining SPAR chunks, then verify gzip integrity.
set -uo pipefail
cd /home/c30084464/Documents/code/SpatialStack_OPSD
REV=976c19177468eabe64e9e2dd0f0450cd32dacc1f
BASE="https://hf-mirror.com/datasets/jasonzhango/SPAR-7M/resolve/$REV"
EXPECTED=10737418240
OUT=./data/media/spar
LOG=./output/spar_chunks_dl.log
exec >> "$LOG" 2>&1
echo "=== spar aria2c-resume start $(date) ==="

dl_one() {
  local c=$1
  local f="$OUT/spar-$c.tar.gz"
  for attempt in $(seq 1 50); do
    sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$sz" -ge "$EXPECTED" ]; then echo "[aria-resume] spar-$c complete ($sz)"; return 0; fi
    echo "[aria-resume] spar-$c attempt $attempt (cur=$sz) at $(date)"
    aria2c -c -x8 -s8 -k1M --file-allocation=none --max-tries=0 --retry-wait=3 \
      --console-log-level=warn --summary-interval=0 -d "$OUT" -o "spar-$c.tar.gz" "$BASE/spar-$c.tar.gz" \
      || echo "[aria-resume] spar-$c aria2c rc=$? (retry)"
  done
  sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
  [ "$sz" -ge "$EXPECTED" ] && { echo "[aria-resume] spar-$c complete ($sz)"; return 0; }
  echo "[aria-resume] spar-$c FAILED: $sz"; return 1
}
export -f dl_one; export BASE EXPECTED OUT

for c in 03 05 06; do dl_one "$c" & done
wait
echo "=== aria2c-resume end $(date) ==="
for c in 03 05 06 11; do echo "spar-$c: $(stat -c %s $OUT/spar-$c.tar.gz 2>/dev/null)"; done
echo "=== gzip integrity check ==="
cat $OUT/spar-{00,01,02,03,04,05,06,07,08,09,10,11,12,13}.tar.gz | pigz -dc 2>/tmp/gzerr | wc -c
echo "gzerr:"; cat /tmp/gzerr | head -5
