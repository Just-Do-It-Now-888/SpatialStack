#!/usr/bin/env bash
# Parallel curl --range downloader for spar-06, with 16MB sub-range fallback for
# throttled ranges whose connection dies mid-download.
set -uo pipefail
cd /home/c30084464/Documents/code/SpatialStack_OPSD
REV=976c19177468eabe64e9e2dd0f0450cd32dacc1f
MIRROR="https://hf-mirror.com/datasets/jasonzhango/SPAR-7M/resolve/$REV"
EXPECTED=10737418240
OUT=./data/media/spar
RANGE_SIZE=134217728   # 128MB
SUB_SIZE=16777216      # 16MB
PARALLEL=16
LOG=./output/spar_chunks_dl.log
exec >> "$LOG" 2>&1

dl_range_sub() {
  # download [start,end] in 16MB sub-ranges into part file (reliable for throttled ranges)
  local c=$1 start=$2 end=$3
  local part="$OUT/spar-$c.part.$start"
  local want=$((end-start+1))
  : > "$part"
  local s=$start
  while [ $s -le $end ]; do
    local e=$((s+SUB_SIZE-1)); [ $e -gt $end ] && e=$end
    local w=$((e-s+1)) got=0
    for try in 1 2 3 4 5 6 7 8; do
      curl -sL -r $s-$e --connect-timeout 20 --retry 20 --retry-delay 2 --max-time 120 \
        -o "$part.subtmp" "$MIRROR/spar-$c.tar.gz" 2>/dev/null
      got=$(stat -c %s "$part.subtmp" 2>/dev/null || echo 0)
      [ "$got" -eq "$w" ] && break
    done
    if [ "$got" -ne "$w" ]; then echo "[sub] FAIL spar-$c $s-$e ($got/$w)"; return 1; fi
    cat "$part.subtmp" >> "$part"
    s=$((e+1))
  done
  rm -f "$part.subtmp"
  got=$(stat -c %s "$part" 2>/dev/null || echo 0)
  [ "$got" -eq "$want" ]
}

dl_range() {
  local c=$1 start=$2 end=$3
  local part="$OUT/spar-$c.part.$start"
  local expected_len=$((end-start+1))
  local psz
  psz=$(stat -c %s "$part" 2>/dev/null || echo 0)
  if [ "$psz" -eq "$expected_len" ]; then return 0; fi
  for try in 1 2 3; do
    curl -sL -r $start-$end --connect-timeout 20 --retry 10 --retry-delay 2 --max-time 300 \
      -o "$part" "$MIRROR/spar-$c.tar.gz" 2>/dev/null
    psz=$(stat -c %s "$part" 2>/dev/null || echo 0)
    [ "$psz" -eq "$expected_len" ] && return 0
  done
  # fallback to 16MB sub-ranges
  echo "[range] spar-$c $start-$end falling back to sub-ranges $(date)"
  dl_range_sub $c $start $end
}
export -f dl_range dl_range_sub; export MIRROR OUT SUB_SIZE

c=06
total=$EXPECTED
rm -f "$OUT/spar-$c".part.* "$OUT/spar-$c.new.tar.gz" "$OUT/spar-$c.subtmp"
n=$(( (total + RANGE_SIZE - 1) / RANGE_SIZE ))
echo "[chunk] spar-$c start $(date): $n ranges x $RANGE_SIZE bytes, $PARALLEL parallel"
seq 0 $((n-1)) | awk -v rs=$RANGE_SIZE -v tot=$total -v c=$c '{s=$1*rs; e=s+rs-1; if(e>=tot)e=tot-1; print c" "s" "e}' \
  | xargs -P $PARALLEL -L1 bash -c 'dl_range "$@"' _
# verify all parts
missing=0 i=0
while [ $i -lt $n ]; do
  local_s=$((i*RANGE_SIZE)); local_e=$((local_s+RANGE_SIZE-1)); [ $local_e -ge $total ] && local_e=$((total-1))
  local_want=$((local_e-local_s+1)); local_got
  local_got=$(stat -c %s "$OUT/spar-$c.part.$local_s" 2>/dev/null || echo 0)
  [ "$local_got" -eq "$local_want" ] || { echo "[chunk] BAD spar-$c part $local_s ($local_got/$local_want)"; missing=1; }
  i=$((i+1))
done
[ $missing -eq 1 ] && { echo "[chunk] spar-$c incomplete $(date)"; exit 1; }
i=0
while [ $i -lt $n ]; do
  cat "$OUT/spar-$c.part.$((i*RANGE_SIZE))" 2>/dev/null
  i=$((i+1))
done > "$OUT/spar-$c.new.tar.gz"
sz=$(stat -c %s "$OUT/spar-$c.new.tar.gz" 2>/dev/null || echo 0)
if [ "$sz" -eq "$EXPECTED" ]; then
  mv "$OUT/spar-$c.new.tar.gz" "$OUT/spar-$c.tar.gz"
  rm -f "$OUT/spar-$c".part.*
  echo "[chunk] spar-$c complete ($sz) $(date)"
else
  echo "[chunk] spar-$c concat size mismatch $sz != $EXPECTED $(date)"
  exit 1
fi
