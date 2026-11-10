#!/bin/bash
# One training-dump job of a train chain for the calibrations of TICLv6 (p4dev area): RECO v6 from the chain's DIGI on
# NFS with the study probe TICLCalibrationProbe (charged-hadron groups on a final trackster, TICL e/gamma superclusters)
# and the adaptive trackster associator without the pT floor.
# usage: run_cal.sh <tt|eg|qcd> <chain> <cpus>; NEV (default 50). Output calib/<sample>/<chain>; marker done or failed.
set -uo pipefail
sample=$1; chain=$(printf "%06d" "$((10#$2))"); cpus=$3
C=/data/fpantale/ticlv6retrain/calib
AREA=/home/fpantale/2026/ticlv6/p4dev/CMSSW_20_1_0_pre4
d=$C/$sample/$chain
case $sample in
  tt) tmpl=/data/fpantale/ticlv6prod/tt/$chain/reco_v6/cfg.py; digi=/cms-hlt-nfs/user/fpantale/ticlv6prod/tt/$chain/digi/step2.root ;;
  eg) tmpl=/data/fpantale/ticlv6prod/eg/000000/reco_v6/cfg.py; digi=/cms-hlt-nfs/user/fpantale/ticlv6prod_c2/eg/$chain/digi/step2.root ;;
  qcd) tmpl=/data/fpantale/ticlv6retrain/jets/qcd/000000/reco_v6/cfg.py; digi=/cms-hlt-nfs/user/fpantale/ticlv6prod_c2/qcd/$chain/digi/step2.root ;;
  *) exit 2 ;;
esac
rm -rf "$d"; mkdir -p "$d" || exit 3
[ -f "$digi" ] || { echo "no DIGI $digi" > "$d/failed"; exit 1; }
[ -f "$tmpl" ] || { echo "no template $tmpl" > "$d/failed"; exit 1; }
{ grep -v "dumpHypotheses = True\|+= process.hypDumper" "$tmpl"; echo; echo "# calibration block"
  echo "process.source.fileNames = cms.untracked.vstring('file:$digi')"
  echo "process.maxEvents.input = ${NEV:-50}"
  cat <<'PY'
process.options.wantSummary = True
if hasattr(process, "hypDumper"):
    delattr(process, "hypDumper")
process.truthBranchTargetsNoFloor = process.truthBranchTargets.clone()
process.truthBranchTargetsNoFloor.branchSelector.ptMin = 0.0
process.truthBranchTracksterAssociatorsNoFloor = process.truthBranchTracksterAssociators.clone(
    recoCollections=["ticlTracksterInterpretations"],
    targetsSrc=("truthBranchTargetsNoFloor", "selectedRoots"),
    assignableTargetsSrc=("truthBranchTargetsNoFloor", "assignableRoots"))
process.calibProbe = cms.EDAnalyzer("TICLCalibrationProbe",
    tracksters=cms.InputTag("ticlTracksterInterpretations"),
    trackToTrackster=cms.InputTag("ticlTracksterInterpretations", "trackToTrackster"),
    trackMode=cms.InputTag("ticlTracksterInterpretations", "trackMode"),
    tracks=cms.InputTag("generalTracks"),
    trackRecoToTruth=cms.InputTag("allTrackToTruthBranchAssociators", "generalTracksRecoToTruthFixed"),
    tracksterRecoToTruth=cms.InputTag("truthBranchTracksterAssociatorsNoFloor", "ticlTracksterInterpretationsRecoToTruthAdaptiveNominal"),
    graph=cms.InputTag("truthLogicalGraphProducer"),
    superClusters=cms.InputTag("ticlEGammaSuperClusters"),
    eGammaTracksters=cms.InputTag("ticlEGammaTracksters"),
    genParticles=cms.InputTag("genParticles"),
    trackMomentumErrorScale=cms.double(2.0))
process.truthBranchValidationPath += process.truthBranchTargetsNoFloor + process.truthBranchTracksterAssociatorsNoFloor + process.calibProbe
PY
} > "$d/cfg.py"
cd $AREA/src && source /cvmfs/cms.cern.ch/cmsset_default.sh && eval "$(scram runtime -sh)" || exit 3
cd "$d" && numactl --physcpubind="$cpus" cmsRun cfg.py > job.log 2>&1
rc=$?
n=$(sed -n 's/^TrigReport Events total = \([0-9]*\).*/\1/p' job.log | head -1)
[ $rc -eq 0 ] && [ "${n:-0}" = "${NEV:-50}" ] || { echo "RC=$rc events=${n:-none}" > failed; exit 1; }
echo "RC=0 events=$n" > done
