#!/bin/bash
# One A/B leg of the TICLv6 baseline validation on one chain of the ticlv6 A/B set, in cmssw-el8: RECO from the DIGI
# of the chain with the template and the CMSSW area of the leg ($VBF_BASE/legs.conf, lines "<leg> <area> <template of
# production/cfg> [parameter block]"; the parameter block is python appended last), and the job block of the chain's production configuration: the probes candProbe and jetMETProbe
# (CornettoStudy/Probe), the truth validation DQM and the truth preset of the sample. The block drops the call of
# customiseTruthMixedReco.customiseValidation (it does not exist with the truth graph of PR 51829), the hypothesis
# dump and the interpretation model settings of the e/gamma blocks (v2 models): the leg sets the models. The TICLv5 leg (template reco_v5.py) gets the block with _v6 = False and without its TICLv6 section.
# Outputs in $VBF_OUT/<kind>/<leg>/<chain>: cfg.py, job.log, time.txt, ntuple.root, dqm.root, rev.root (TICL candidates,
# final tracksters, assignment maps, pfTICL, linked tracksters), egsc.root for e/gamma; the marker done last.
# kind: tt (TenTau), eg (e/gamma guns), qcd (QCD), qcdhi (high-pT QCD, with the QCD block of chain 0).
# usage: ticl_leg.sh <kind> <chain> <leg>, or ticl_leg.sh <kind>-<leg> <chain> (the form of a pool_vbf.sh queue line);
# environment: VBF_BASE, VBF_OUT, VBF_WORK; optional VBF_THREADS (16), VBF_EXTRA_CFG (python appended to the job)
# and VBF_FILES (the files the job publishes, separated by spaces).
set -uo pipefail
X=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ $# -eq 2 ]; then
  kind=${1%%-*}; leg=${1#*-}; chain=$(printf "%06d" "$2")
else
  kind=$1; chain=$(printf "%06d" "$2"); leg=$3
fi
N=${DATA:-/cms-hlt-nfs/user/fpantale}
case $kind in
  tt) src=$N/ticlv6prod/tt/$chain/reco_v6/cfg.py; digi=$N/ticlv6prod/tt/$chain/digi/step2.root ;;
  eg) src=$N/ticlv6prod/eg/$chain/reco_v6/cfg.py; digi=$N/ticlv6prod/eg/$chain/digi/step2.root ;;
  qcd) src=$N/ticlv6retrain/jets/qcd/$chain/reco_v6/cfg.py; digi=$N/ticlv6retrain/jets/qcd/$chain/digi/step2.root ;;
  qcdhi) src=$N/ticlv6retrain/jets/qcd/000000/reco_v6/cfg.py; digi=$N/ticlv6prod_c3/qcdhi/$chain/digi/step2.root ;;
  *) echo "unknown kind $kind" >&2; exit 2 ;;
esac
read -r _ AREA TEMPLATE PARAMS <<< "$(awk -v l="$leg" '$1 == l' "${VBF_BASE:?VBF_BASE}/legs.conf")"
[ -n "${AREA:-}" ] && [ -f "$X/common/cfg/$TEMPLATE" ] || { echo "leg $leg not in legs.conf" >&2; exit 2; }
[ -z "${PARAMS:-}" ] || [ -f "$PARAMS" ] || { echo "no parameter block $PARAMS" >&2; exit 2; }
[ -f "$src" ] && [ -s "$digi" ] || { echo "no block $src or DIGI $digi" >&2; exit 4; }
OUT=${VBF_OUT:?VBF_OUT}/$kind/$leg/$chain
W=${VBF_WORK:?VBF_WORK}/$kind/$leg/$chain
THREADS=${VBF_THREADS:-16}
[ -f "$OUT/done" ] && exit 0
rm -rf "$W" "$OUT"; mkdir -p "$W" "$OUT" || exit 3
cp "$digi" "$W/.step2.root.tmp" && [ "$(stat -L -c %s "$digi")" = "$(stat -c %s "$W/.step2.root.tmp")" ] &&
  mv "$W/.step2.root.tmp" "$W/step2.root" || { echo "DIGI copy failed" >&2; exit 5; }
cd "$AREA/src" && source /cvmfs/cms.cern.ch/cmsset_default.sh && eval "$(SCRAM_IGNORE_HOOKS=1 scram runtime -sh)" || exit 3
code=$(cat "$AREA/code_id.txt" 2>/dev/null || echo "$AREA")
[ -n "${PARAMS:-}" ] && code="$code params:$(md5sum < "$PARAMS" | cut -c1-12)"
cd "$W" || exit 3
{
  cat "$X/common/cfg/$TEMPLATE"
  echo; echo "# Job block of $src (without customiseValidation, the hypothesis dump and model settings)"
  if [ "$TEMPLATE" = reco_v5.py ]; then
    sed -n '/^# Job block/,$p' "$src" | sed 's/^_v6 = True$/_v6 = False/' | sed '/^# TICLv6 only/,/+= process.hypDumper$/d'
  else
    sed -n '/^# Job block/,$p' "$src" | grep -v 'dumpHypotheses = True\|+= process.hypDumper'
  fi | grep -v 'customiseTruthMixedReco import customiseValidation\|^customiseValidation(process)' |
    grep -v '^# Interpretation models v2\|^process.ticlTracksterInterpretations.\(arbitrationModelFile\|arbitrationPlattA\|arbitrationPlattB\|neutralModelFile\) *='
  echo; echo "# Leg block: $kind $chain $leg"
  cat <<PY
process.source.fileNames = cms.untracked.vstring("file:$W/step2.root")
process.maxEvents.input = -1
process.options.numberOfThreads = $THREADS
process.options.numberOfStreams = $((THREADS / 2))
process.options.wantSummary = True
if hasattr(process, "hypDumper"):
    delattr(process, "hypDumper")
process.revOut = cms.OutputModule("PoolOutputModule", fileName = cms.untracked.string("rev.root"),
    outputCommands = cms.untracked.vstring("drop *", "keep *_ticlCandidate_*_*", "keep *_ticlTracksterInterpretations_*_*",
                                           "keep *_pfTICL_*_*", "keep *_ticlTracksterLinks_*_*"))
process.revOutPath = cms.EndPath(process.revOut)
process.schedule.append(process.revOutPath)
PY
  if [ -n "${VBF_EXTRA_CFG:-}" ]; then echo; echo "# Extra block: $VBF_EXTRA_CFG"; cat "$VBF_EXTRA_CFG"; fi
  if [ -n "${PARAMS:-}" ]; then echo; echo "# Parameter block: $PARAMS"; cat "$PARAMS"; fi
  if [ "$kind" = eg ]; then cat <<'PY'
process.egscOut = cms.OutputModule("PoolOutputModule", fileName = cms.untracked.string("egsc.root"),
    outputCommands = cms.untracked.vstring("drop *", "keep *_genParticles_*_*",
        "keep *_particleFlowSuperClusterHGCal_*_*", "keep *_ticlEGammaSuperClusters_*_*", "keep *_ticlEGammaTracksters_*_*",
        "keep *_ecalDrivenElectronSeeds_*_*", "keep *_electronGsfTracks_*_*", "keep *_ecalDrivenGsfElectronsHGC_*_*",
        "keep *_ecalDrivenGsfElectronCoresHGC_*_*", "keep *_photonsHGC_*_*", "keep *_photonCoreHGC_*_*",
        "keep *_ticlCandidate_*_*", "keep *_ticlElectronSeeds_*_*", "keep *_ticlElectronSuperClusters_*_*"))
process.egscOutPath = cms.EndPath(process.egscOut)
process.schedule.append(process.egscOutPath)
PY
  fi
} > cfg.py
s=$(date +%s)
/usr/bin/time -f "MAXRSS_KB=%M" -o time.txt cmsRun cfg.py > job.log 2>&1
rc=$?
echo "SECONDS=$(( $(date +%s) - s ))" >> time.txt
n=$(sed -n 's/^TrigReport Events total = \([0-9]*\).*/\1/p' job.log | head -1)
[ $rc -eq 0 ] && [ -z "$n" ] && rc=65
files=${VBF_FILES:-"cfg.py job.log time.txt ntuple.root dqm.root rev.root"}
[ "$kind" = eg ] && [ -z "${VBF_FILES:-}" ] && files="$files egsc.root"
if [ $rc -eq 0 ]; then
  for f in $files; do
    cp "$f" "$OUT/.$f.tmp" && mv "$OUT/.$f.tmp" "$OUT/$f" && [ "$(stat -c %s "$f")" = "$(stat -c %s "$OUT/$f")" ] || exit 6
  done
  echo "RC=0 events=$n code=$code" > "$OUT/done"
else
  cp cfg.py job.log time.txt "$OUT/" 2>/dev/null
  echo "RC=$rc events=${n:-none} code=$code" > "$OUT/failed"
fi
rm -rf "$W"
exit $rc
