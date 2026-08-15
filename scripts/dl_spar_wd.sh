#!/usr/bin/env bash
# Watchdog download for 05/06: curl -C - with --max-time 120, loop until correct size. Never hangs, never loses progress.
set -uo pipefail
cd /home/c30084464/Documents/code/SpatialStack_OPSD
REV=976c19177468eabe64e9e2dd0f0450cd32dacc1f
BASE="https://hf-mirror.com/datasets/jasonzhango/SPAR-7M/resolve/$REV"
EXPECTED=10737418240
OUT=./data/media/spar
LOG=./output/spar_chunks_dl.log
exec >> "$LOG" 2>&1
echo "=== watchdog 05/06 start $(date) ==="

dl_wd() {
  local c=$1 f="$OUT/spar-$c.tar.gz"
  for attempt in $(seq 1 1000); do
    sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$sz" -ge "$EXPECTED" ]; then echo "[wd] spar-$c complete ($sz)"; return 0; fi
    # each curl runs at most 120s, then exits; loop resumes with -C -
    curl -sL -C - --connect-timeout 20 --max-time 120 --retry 5 --retry-delay 3 -o "$f" "$BASE/spar-$c.tar.gz" >/dev/null 2>&1
  done
  sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
  [ "$sz" -ge "$EXPECTED" ] && { echo "[wd] spar-$c complete ($sz)"; return 0; }
  echo "[wd] spar-$c FAILED: $sz"; return 1
}
export -f dl_wd; export BASE EXPECTED OUT

dl_wd 05 &
dl_wd 06 &
wait
echo "=== watchdog 05/06 end $(date) ==="
for c in 05 06 11 03; do echo "spar-$c: $(stat -c %s $OUT/spar-$c.tar.gz 2>/dev/null)"; done
