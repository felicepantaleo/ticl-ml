#!/bin/bash
# Mixture-PID retrain of TAG, in felice-1gpu-nvl inside cmssw-el8 --nv: waits for the probe chains of the queue (deadline
# 6 h; stops on a failed chain), extracts ts_flat.root, trains hi (CLUEstering tracksters) and fin (final tracksters) with
# the H9 recipe and maps them onto the CNN scale. Output <TAG>/models/mix/mixPID_{hi,fin}_<TAG>*; marker <TAG>/mix.done or mix.failed.
# usage: RUNDIR=<run folder> TAG=rt3p2 [TRAIN_ENV=<cmssw area with torch>] train_mix.sh
X=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
W=${RUNDIR:?RUNDIR}
D=$W/${TAG:?TAG}
L=$X/showerShapePID/pidstudy
O=$D/models/mix
fail() { echo "$1" > $D/mix.failed; exit 1; }
rm -f $D/mix.done $D/mix.failed
need=$(grep -c . $D/queue_${TAG}_probe.txt)
end=$(( $(date +%s) + 21600 ))
until [ "$(ls -d $D/probe/*/*/done 2>/dev/null | wc -l)" -ge "$need" ]; do
  [ "$(ls -d $D/probe/*/*/failed 2>/dev/null | wc -l)" = 0 ] || fail "failed probe chains"
  [ "$(date +%s)" -lt "$end" ] || fail "probe chains not done after 6 h"; sleep 60
done
cd ${TRAIN_ENV:-/shared/fpantale/retrain_cs/CMSSW_20_1_0_pre4}/src && source /cvmfs/cms.cern.ch/cmsset_default.sh && eval "$(SCRAM_IGNORE_HOOKS=1 scram runtime -sh)" || fail cmsenv
ls -d $D/probe/*/*/done | xargs -n1 dirname | xargs -P 16 -I{} sh -c '[ -f {}/ts_flat.root ] || python3 -I '$L'/stage2/extract.py {} > {}/extract.log 2>&1'
echo "$(date '+%F %T') extracted $(ls $D/probe/*/*/ts_flat.root | wc -l)"
cd $O || fail "no $O"
export PROBE_DIR=$D/probe
( python3 -u $L/mix/train_mix.py hi $O/mixPID_hi_$TAG > $O/train_hi.log 2>&1 && python3 -u $L/mix/export_mix.py hi $O/mixPID_hi_$TAG > $O/export_hi.log 2>&1 ) || fail "hi" &
( python3 -u $L/mix/train_mix.py fin $O/mixPID_fin_$TAG > $O/train_fin.log 2>&1 && python3 -u $L/mix/export_mix.py fin $O/mixPID_fin_$TAG > $O/export_fin.log 2>&1 ) || fail "fin" &
wait
[ -f $D/mix.failed ] && exit 1
md5sum $O/*_mapped.onnx > $O/md5sums.txt
echo "$(date '+%F %T') done" > $D/mix.done
