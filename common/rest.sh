#!/bin/bash
# The steps of TAG after the model training, on patatrack-bg-01: waits for <TAG>/train/train.done (deadline 30 h),
# installs the mixture PIDs and the interpretation models into AREA, runs the 148 supercluster legs (32 slots of
# 16 CPUs), trains and installs the supercluster score and type models, writes blocks/models_sc_<TAG>.py, and runs the
# validation leg <POINT>rt on the 96 validation files. Marker <TAG>/rest.done or rest.failed.
# With SC_REF="<reference score onnx> <cut below 10 GeV> <cut at or above 10 GeV>", the score cuts of the new model are set
# at the fake count of the reference (equal_fake_cuts.py) and written into blocks/models_sc_<TAG>.py.
# usage: RUNDIR=<run folder> AREA=<cmssw area> TAG=rt3p2 rest.sh; optional SC_REF, SC_SLOTS ("<cpus> <membind>,..."),
# VALDIR (folder of legs.conf and of the validation driver), VALRUN (validation driver, default run_val.sh), VALP (its
# output folder).
X=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
W=${RUNDIR:?RUNDIR}
D=$W/${TAG:?TAG}
source $D/point.env || exit 1
A=${AREA:?AREA}
V=${VALDIR:-/home/fpantale/2026/ticlv6/hackathon/vtx}
VP=${VALP:-/data/fpantale/hackathon_vtx/prod_val_$TAG}
fail() { echo "$(date '+%F %T') $1" > $D/rest.failed; exit 1; }
rm -f $D/rest.done $D/rest.failed
end=$(( $(date +%s) + 108000 ))
until [ -f $D/train/train.done ]; do
  ls $D/train/*.failed > /dev/null 2>&1 && fail "model training failed"
  [ "$(date +%s)" -lt "$end" ] || fail "no train.done after 30 h"
  sleep 300
done
P=$A/src/RecoTICL/Inference/data/showerShapePID
for l in hi fin; do
  cp $D/models/mix/mixPID_${l}_${TAG}_mapped.onnx $P/.mixPID_${l}_$TAG.onnx.tmp && mv $P/.mixPID_${l}_$TAG.onnx.tmp $P/mixPID_${l}_$TAG.onnx || fail "install mixPID"
done
bash $X/common/install_models.sh $A || fail "install models"
date '+%F %T' > $D/models.installed
slots=${SC_SLOTS:-$(for s in $(seq 0 16 496); do printf "%s-%s %s," $s $((s + 15)) $(( s < 128 || (s >= 256 && s < 384) ? 0 : 1 )); done)}
VBF_WORK=/dev/shm/fpantale/${TAG}_sc bash $X/egammaSuperClusters/score_type/run_sc.sh "${slots%,}" || fail "supercluster legs"
grep -q " 0 failed" $D/sc/ALL_DONE || fail "supercluster legs: $(cat $D/sc/ALL_DONE)"
AREA=$A CPUS=0-63 bash $X/egammaSuperClusters/score_type/train_sc.sh || fail "supercluster training"
S=$A/src/RecoTICL/ParticleFlow/data/egammaSuperClusters
cp $D/models/sc/score/score.onnx $S/.score_$TAG.onnx.tmp && mv $S/.score_$TAG.onnx.tmp $S/score_$TAG.onnx || fail "install score"
cp $D/models/sc/type/score.onnx $S/.type_$TAG.onnx.tmp && mv $S/.type_$TAG.onnx.tmp $S/type_$TAG.onnx || fail "install type"
cat > $W/blocks/models_sc_$TAG.py <<PY
# The supercluster score and type models retrained on the $POINT tracksters ($TAG); the energy regression stays regression_v3.
process.ticlEGammaSuperClusters.scoreModel = 'RecoTICL/ParticleFlow/data/egammaSuperClusters/score_$TAG.onnx'
process.ticlEGammaSuperClusters.typeScoreModel = 'RecoTICL/ParticleFlow/data/egammaSuperClusters/type_$TAG.onnx'
PY
if [ -n "${SC_REF:-}" ]; then
  read -r ref rlo rhi <<< "$SC_REF"
  cuts=$(cd $A/src && source /cvmfs/cms.cern.ch/cmsset_default.sh && eval "$(SCRAM_IGNORE_HOOKS=1 scram runtime -sh)" &&
    python3 -I $X/egammaSuperClusters/score_type/equal_fake_cuts.py $D/sc/legs/eg sc_$TAG $ref $rlo $rhi $S/score_$TAG.onnx |
    tee $D/models/sc/equal_fake_cuts.txt | awk '$1 == "CUTS" {print $2, $3}')
  read -r clo chi <<< "$cuts"
  [ -n "$chi" ] || fail "equal-fake cuts"
  cat >> $W/blocks/models_sc_$TAG.py <<PY
# Score cuts at the fake count of $(basename $ref) at $rlo and $rhi (validation chains 90-99, $D/models/sc/equal_fake_cuts.txt).
process.ticlEGammaSuperClusters.minScore = $clo
process.ticlEGammaSuperClusters.minScoreHighEt = $chi
PY
fi
grep -q "^${POINT}rt " $V/legs.conf || echo "${POINT}rt $A reco_v6.py $W/blocks/val_${POINT}rt.py" >> $V/legs.conf
cd $V && LEGS=${POINT}rt P=$VP bash ${VALRUN:-run_val.sh} || fail "validation"
grep -q " 0 failed" $VP/ALL_DONE || fail "validation: $(cat $VP/ALL_DONE)"
echo "$(date '+%F %T') done" > $D/rest.done
