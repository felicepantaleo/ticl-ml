#!/bin/bash
# One training-dump job of the CLUEstering retrain (H9), job2.sh with the study block from BLOCK (default cfg/<mode>_block.py): RECO of one chain's DIGI with the cs
# configuration (cfg/reco_v6.py) and one study block (cfg/<mode>_block.py), see bin/job.sh. The DIGI is copied to
# WORK first, so cmsRun reads only local files (parallel reads from NFS hung the NFS client of felice-vbf-hc).
# usage: job3.sh <probe|m6> <tt|eg|qcd|qcdhi> <chain> <cpus>; AREA, WORK, OUTROOT (default $N/<mode>new), NEV (50).
# Output $OUTROOT/<sample>/<chain>/{cfg.py,job.log,ntuple.root[,tracksters.root]} and done or failed.
set -uo pipefail
mode=$1; sample=$2; chain=$(printf "%06d" "$((10#$3))"); cpus=$4
N=/cms-hlt-nfs/user/fpantale/hackathon_retrain
NF=/cms-hlt-nfs/user/fpantale
PRESET=""
case $sample in
  tt) digi=$NF/ticlv6prod/tt/$chain/digi/step2.root
      PRESET='applyTruthPreset(process, fragment="TenTau_E15to500_Eta3p1_pythia8_cfi", keepStableSpectators=True)' ;;
  eg) digi=$NF/ticlv6prod_c2/eg/$chain/digi/step2.root ;;
  qcd) digi=$NF/ticlv6prod_c2/qcd/$chain/digi/step2.root ;;
  qcdhi) digi=$NF/ticlv6prod_c3/qcdhi/$chain/digi/step2.root ;;
  *) exit 2 ;;
esac
case $mode in probe|m6) ;; *) exit 2 ;; esac
OUT=${OUTROOT:-$N/${mode}new}/$sample/$chain
W=${WORK:?WORK}/$mode/$sample/$chain
[ -f "$OUT/done" ] && exit 0
rm -rf "$OUT" "$W"; mkdir -p "$OUT" "$W" || exit 3
[ -s "$digi" ] || { echo "no DIGI $digi" > "$OUT/failed"; exit 1; }
# At most COPYSLOTS (default 8) DIGI copies from NFS at the same time on this host: one flock per slot.
copy_digi() {
  local k end=$(( $(date +%s) + 3600 ))
  while [ "$(date +%s)" -lt "$end" ]; do
    for k in $(seq 0 $(( ${COPYSLOTS:-8} - 1 ))); do
      if flock -n "${WORK}/.copy.$k" -c "timeout 900 cp '$digi' '$W/.step2.root.tmp'"; then return 0; fi
      [ -f "$W/.step2.root.tmp" ] && return 1
    done
    sleep 10
  done
  return 1
}
copy_digi && [ "$(stat -L -c %s "$digi")" = "$(stat -c %s "$W/.step2.root.tmp")" ] &&
  mv "$W/.step2.root.tmp" "$W/step2.root" || { echo "DIGI copy failed" > "$OUT/failed"; rm -rf "$W"; exit 5; }
nthr=$(echo "$cpus" | awk -F'[-,]' '{print ($2 == "" ? 1 : $2 - $1 + 1)}')
{
  cat $N/cfg/reco_v6.py
  cat <<PY

# Job block: $mode $sample $chain
process.source.fileNames = cms.untracked.vstring("file:$W/step2.root")
process.maxEvents.input = ${NEV:-50}
process.options.numberOfThreads = $nthr
process.options.numberOfStreams = $nthr
process.options.wantSummary = True
process.schedule.remove(process.AODSIMoutput_step)
del process.AODSIMoutput_step, process.AODSIMoutput
process.schedule.remove(process.DQMoutput_step)
del process.DQMoutput_step, process.DQMoutput
if not hasattr(process, "DQMStore"):
    process.load("DQMServices.Core.DQMStoreNonLegacy_cff")
from SimGeneral.TruthGraphAssociatorProducers.customiseTruthGraphAssociators import customiseTruthBranchValidation
customiseTruthBranchValidation(process)
from PhysicsTools.TruthInfo.customiseTruthPreset import applyTruthPreset
$PRESET
process.TFileService = cms.Service("TFileService", fileName=cms.string("ntuple.root"))
PY
  cat "${BLOCK:-$N/cfg/${mode}_block.py}"
} > "$W/cfg.py"
cd "${AREA:?AREA}/src" && source /cvmfs/cms.cern.ch/cmsset_default.sh && eval "$(SCRAM_IGNORE_HOOKS=1 scram runtime -sh)" || exit 3
export CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=
cd "$W" && timeout 7200 taskset -c "$cpus" cmsRun cfg.py > job.log 2>&1
rc=$?
n=$(sed -n 's/^TrigReport Events total = \([0-9]*\).*/\1/p' job.log | head -1)
[ $rc -eq 0 ] && [ "${n:-0}" != "${NEV:-50}" ] && rc=65
if [ $rc -eq 0 ]; then
  for f in cfg.py job.log ntuple.root tracksters.root; do
    [ -f "$f" ] || continue
    cp "$f" "$OUT/.$f.tmp" && mv "$OUT/.$f.tmp" "$OUT/$f" || { echo "copy $f" > "$OUT/failed"; exit 6; }
  done
  echo "RC=0 events=$n area=$AREA" > "$OUT/done"
else
  cp cfg.py job.log "$OUT/" 2>/dev/null
  echo "RC=$rc events=${n:-none}" > "$OUT/failed"
fi
rm -rf "$W"
exit $rc
