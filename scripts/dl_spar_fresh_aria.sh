#!/usr/bin/env bash
# Fresh aria2c (delete first, no -c) with fresh signed URL + --max-tries=0 (no 403 gaps), 300s timeout, retry until complete.
set -uo pipefail
cd /home/c30084464/Documents/code/SpatialStack_OPSD
REV=976c19177468eabe64e9e2dd0f0450cd32dacc1f
MIRROR="https://hf-mirror.com/datasets/jasonzhango/SPAR-7M/resolve/$REV"
EXPECTED=10737418240
OUT=./data/media/spar
LOG=./output/spar_chunks_dl.log
exec >> "$LOG" 2>&1
echo "=== fresh-aria2c 05/06 start $(date) ==="

dl_fresh() {
  local c=$1 f="$OUT/spar-$c.tar.gz"
  for attempt in $(seq 1 20); do
    sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$sz" -eq "$EXPECTED" ]; then echo "[fresh] spar-$c complete ($sz)"; return 0; fi
    # delete for true fresh download (avoid -c control-file mismatch)
    rm -f "$f" "$f.aria2"
    # fetch fresh signed URL each attempt
    local signed=$(curl -sI -o /dev/null -w "%{redirect_url}" --max-time 20 "$MIRROR/spar-$c.tar.gz")
    echo "[fresh] spar-$c attempt $attempt at $(date) (signed len ${#signed})"
    aria2c -x16 -s16 -k1M --file-allocation=none --max-tries=0 --retry-wait=2 \
      --console-log-level=warn --summary-interval=0 --timeout=60 --connect-timeout=20 \
      -d "$OUT" -o "spar-$c.tar.gz" "$signed" 2>&1 | tail -2
  done
  sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
  [ "$sz" -eq "$EXPECTED" ] && { echo "[fresh] spar-$c complete ($sz)"; return 0; }
  echo "[fresh] spar-$c FAILED: $sz"; return 1
}
export -f dl_fresh; export MIRROR EXPECTED OUT
dl_fresh 05 &
dl_fresh 06 &
wait
echo "=== fresh-aria2c 05/06 end $(date) ==="
for c in 05 06; do echo "spar-$c: $(stat -c %s $OUT/spar-$c.tar.gz 2>/dev/null)"; done
echo "=== verify full archive ==="
cat $OUT/spar-{00,01,02,03,04,05,06,07,08,09,10,11,12,13}.tar.gz | gzip -dc 2>/tmp/gzv | wc -c
echo "verify errors:"; grep -c "invalid\|corrupt\|unexpected" /tmp/gzv; head -2 /tmp/gzv
