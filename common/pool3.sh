#!/bin/bash
# Pool of tools/job3.sh workers (inside cmssw-el8). Worker k owns CPUs <first>+k*<per> .. +<per>-1 and claims queue
# lines ("<mode> <sample> <chain>") with an atomic mkdir in <RUNDIR>/claims/<queue>/. A STOP file next to the queue stops
# new claims; a worker stops after 5 failures in a row.
# usage: pool.sh <queue> <first cpu> <workers> <cpus per worker>; AREA, WORK, OUTROOT as for job.sh. Marker <queue>.pool.done.
set -uo pipefail
Q=$1; first=$2; nw=$3; per=$4
X=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
C=${RUNDIR:?RUNDIR}/claims/$(basename "$Q" .txt)
mkdir -p "$C"
rm -f "$Q.pool.done"
worker() {
  local w=$1 lo=$(( first + $1 * per )) fails=0
  local cpus="$lo-$(( lo + per - 1 ))"
  while read -r mode sample chain; do
    [ -z "${mode:-}" ] && continue
    [ -f "$(dirname "$Q")/STOP" ] && break
    mkdir "$C/${mode}_${sample}_${chain}" 2>/dev/null || continue
    bash $X/common/job3.sh "$mode" "$sample" "$chain" "$cpus"; rc=$?
    if [ $rc -eq 0 ]; then fails=0; else fails=$(( fails + 1 )); fi
    echo "$(date +%T) $mode $sample $chain rc=$rc" >> "$Q.log"
    [ $fails -ge 5 ] && { echo "worker $w: 5 failures in a row" >&2; break; }
  done < "$Q"
}
for w in $(seq 0 $(( nw - 1 ))); do worker $w & sleep 2; done
wait
echo "done $(ls -d ${OUTROOT:?OUTROOT}/*/*/done 2>/dev/null | wc -l) failed $(ls -d ${OUTROOT}/*/*/failed 2>/dev/null | wc -l) of $(grep -c . "$Q")" > "$Q.pool.done"
