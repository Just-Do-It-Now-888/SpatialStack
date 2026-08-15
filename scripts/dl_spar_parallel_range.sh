#!/usr/bin/env bash
# Parallel curl --range downloader. Each range request gets a freshly signed URL
# from hf-mirror (206 Partial Content), so no 403 corruption (unlike aria2c) and
# parallel connections bypass the per-IP single-stream throttle.
set -uo pipefail
cd /home/c30084464/Documents/code/SpatialStack_OPSD
REV=976c19177468eabe64e9e2dd0f0450cd32dacc1f
MIRROR="https://hf-mirror.com/datasets/jasonzhango/SPAR-7M/resolve/$REV"
EXPECTED=10737418240
OUT=./data/media/spar
RANGE_SIZE=134217728   # 128MB per range
PARALLEL=16
LOG=./output/spar_chunks_dl.log
mkdir -p output
exec >> "$LOG" 2>&1

dl_range() {
  local c=$1 start=$2 end=$3
  local part="$OUT/spar-$c.part.$start"
  local expected_len=$((end-start+1))
  local psz
  psz=$(stat -c %s "$part" 2>/dev/null || echo 0)
  if [ "$psz" -eq "$expected_len" ]; then return 0; fi
  for try in $(seq 1 15); do
    curl -sL -r $start-$end --connect-timeout 20 --retry 10 --retry-delay 2 --max-time 300 \
      -o "$part" "$MIRROR/spar-$c.tar.gz" 2>/dev/null
    psz=$(stat -c %s "$part" 2>/dev/null || echo 0)
    [ "$psz" -eq "$expected_len" ] && return 0
  done
  echo "[range] FAILED spar-$c $start-$end (got $psz want $expected_len) $(date)"
  return 1
}
export -f dl_range; export MIRROR OUT

dl_chunk() {
  local c=$1 total=$EXPECTED
  rm -f "$OUT/spar-$c".part.* "$OUT/spar-$c.new.tar.gz"
  local n=$(( (total + RANGE_SIZE - 1) / RANGE_SIZE ))
  echo "[chunk] spar-$c start $(date): $n ranges x $RANGE_SIZE bytes, $PARALLEL parallel"
  seq 0 $((n-1)) | awk -v rs=$RANGE_SIZE -v tot=$total -v c=$c '{s=$1*rs; e=s+rs-1; if(e>=tot)e=tot-1; print c" "s" "e}' \
    | xargs -P $PARALLEL -L1 bash -c 'dl_range "$@"' _
  # verify all parts present
  local missing=0 i=0
  while [ $i -lt $n ]; do
    local s=$((i*RANGE_SIZE)); local e=$((s+RANGE_SIZE-1)); [ $e -ge $total ] && e=$((total-1))
    local want=$((e-s+1)); local got
    got=$(stat -c %s "$OUT/spar-$c.part.$s" 2>/dev/null || echo 0)
    [ "$got" -eq "$want" ] || { echo "[chunk] MISSING/BAD spar-$c part $s ($got/$want)"; missing=1; }
    i=$((i+1))
  done
  [ $missing -eq 1 ] && { echo "[chunk] spar-$c incomplete, keeping parts"; return 1; }
  # concatenate in order
  i=0
  while [ $i -lt $n ]; do
    cat "$OUT/spar-$c.part.$((i*RANGE_SIZE))" 2>/dev/null
    i=$((i+1))
  done > "$OUT/spar-$c.new.tar.gz"
  local sz; sz=$(stat -c %s "$OUT/spar-$c.new.tar.gz" 2>/dev/null || echo 0)
  if [ "$sz" -eq "$EXPECTED" ]; then
    mv "$OUT/spar-$c.new.tar.gz" "$OUT/spar-$c.tar.gz"
    rm -f "$OUT/spar-$c".part.*
    echo "[chunk] spar-$c complete ($sz) $(date)"
    return 0
  else
    echo "[chunk] spar-$c concat size mismatch $sz != $EXPECTED $(date)"
    return 1
  fi
}
export -f dl_chunk dl_range; export RANGE_SIZE EXPECTED PARALLEL

echo "=== parallel-range dl start $(date) ==="
dl_chunk 05
dl_chunk 06
echo "=== parallel-range dl end $(date) ==="
