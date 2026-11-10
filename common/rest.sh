#!/bin/bash
# The steps of TAG after the model training, on patatrack-bg-01: waits for <TAG>/train/train.done (deadline 30 h),
# installs the mixture PIDs and the interpretation models into the rt3 area, runs the 148 supercluster legs (16 slots of
# 16 CPUs), trains and installs the supercluster score and type models, writes blocks/models_sc_<TAG>.py, and runs the
# validation leg <POINT>rt on the 96 validation files. Marker <TAG>/rest.done or rest.failed.
# usage: TAG=rt3p2 rest.sh
W=/cms-hlt-nfs/user/fpantale/ticlv6tune/rt3
D=$W/${TAG:?TAG}
source $D/point.env || exit 1
A=/home/fpantale/2026/ticlv6/hackathon/vtx/rt3/CMSSW_20_1_X_2026-11-02-2300
V=/home/fpantale/2026/ticlv6/hackathon/vtx
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
bash $W/install_models.sh $A || fail "install models"
date '+%F %T' > $D/models.installed
slots=$(for s in $(seq 0 16 240); do printf "%s-%s %s," $s $((s + 15)) $((s < 128 ? 0 : 1)); done)
VBF_WORK=/dev/shm/fpantale/${TAG}_sc bash $W/sc/run_sc.sh "${slots%,}" || fail "supercluster legs"
grep -q " 0 failed" $D/sc/ALL_DONE || fail "supercluster legs: $(cat $D/sc/ALL_DONE)"
AREA=$A CPUS=0-63 bash $W/sc/train_sc.sh || fail "supercluster training"
S=$A/src/RecoTICL/ParticleFlow/data/egammaSuperClusters
cp $D/models/sc/score/score.onnx $S/.score_$TAG.onnx.tmp && mv $S/.score_$TAG.onnx.tmp $S/score_$TAG.onnx || fail "install score"
cp $D/models/sc/type/score.onnx $S/.type_$TAG.onnx.tmp && mv $S/.type_$TAG.onnx.tmp $S/type_$TAG.onnx || fail "install type"
cat > $W/blocks/models_sc_$TAG.py <<PY
# The supercluster score and type models retrained on the $POINT tracksters ($TAG); the energy regression stays regression_v3.
process.ticlEGammaSuperClusters.scoreModel = 'RecoTICL/ParticleFlow/data/egammaSuperClusters/score_$TAG.onnx'
process.ticlEGammaSuperClusters.typeScoreModel = 'RecoTICL/ParticleFlow/data/egammaSuperClusters/type_$TAG.onnx'
PY
grep -q "^${POINT}rt " $V/legs.conf || echo "${POINT}rt $A reco_v6.py $W/blocks/val_${POINT}rt.py" >> $V/legs.conf
cd $V && LEGS=${POINT}rt P=/data/fpantale/hackathon_vtx/prod_val_$TAG bash run_val.sh || fail "validation"
grep -q " 0 failed" /data/fpantale/hackathon_vtx/prod_val_$TAG/ALL_DONE || fail "validation: $(cat /data/fpantale/hackathon_vtx/prod_val_$TAG/ALL_DONE)"
echo "$(date '+%F %T') done" > $D/rest.done
