#!/bin/bash
# Calibration training-dump pool over calib/queue.txt (lines "sample chain"): one worker per lane of 4 cores and their
# hyperthreads (c0 + 256). A chain with a done marker is skipped. Deadline 6 h. usage: pool_cal.sh "<first cores>";
# marker calib/pool.done.
C=/data/fpantale/ticlv6retrain/calib
Q=$C/queue.txt
end=$(( $(date +%s) + 21600 ))
worker() {
  local cpus="$1-$(($1 + 3)),$(($1 + 256))-$(($1 + 259))"
  while [ "$(date +%s)" -lt "$end" ]; do
    local line sample chain
    line=$(flock $Q.lock sh -c "head -1 $Q; sed -i 1d $Q")
    [ -z "$line" ] && return
    read -r sample chain <<< "$line"
    [ -f "$C/$sample/$(printf %06d $chain)/done" ] && continue
    "$C/run_cal.sh" "$sample" "$chain" "$cpus"
  done
}
for c0 in $1; do worker $c0 & done
wait
n() { ls -d "$@" 2>/dev/null | wc -l; }
echo "calib eg $(n $C/eg/*/done) tt $(n $C/tt/*/done) qcd $(n $C/qcd/*/done) done, $(n $C/*/*/failed) failed" > $C/pool.done
