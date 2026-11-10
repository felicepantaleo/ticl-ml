#!/bin/bash
# One pool of training-dump jobs of the retrain TAG (tools/pool3.sh of hackathon_retrain, job3.sh, 50 events per chain).
# STEP=probe: queue_<TAG>_probe.txt with blocks/probe_<POINT>.py into <TAG>/probe.
# STEP=dumps: waits for <TAG>/mix.done (deadline 8 h), installs mixPID_{hi,fin}_<TAG>.onnx into AREA, then
#   queue_<TAG>_dumps.txt with blocks/m6s_<POINT>.py into <TAG>/dumps (hypothesis, species and share rows in one pass).
# usage: TAG=rt3p2 STEP=probe|dumps AREA=<area> WORK=<local dir> [COPYSLOTS=n] start_pool.sh <first cpu> <workers> <cpus per worker>
W=/cms-hlt-nfs/user/fpantale/ticlv6tune/rt3
N=/cms-hlt-nfs/user/fpantale/hackathon_retrain
D=$W/${TAG:?TAG}
source $D/point.env || exit 1
: "${AREA:?AREA}" "${WORK:?WORK}"
case "${STEP:?STEP}" in
  probe) OUTROOT=$D/probe BLOCK=$W/blocks/probe_$POINT.py exec bash $N/tools/pool3.sh $D/queue_${TAG}_probe.txt "$@" ;;
  dumps)
    end=$(( $(date +%s) + 28800 ))
    until [ -f $D/mix.done ]; do
      [ -f $D/mix.failed ] && { echo "mix failed: $(cat $D/mix.failed)"; exit 1; }
      [ "$(date +%s)" -lt "$end" ] || { echo "no mix.done after 8 h"; exit 1; }
      sleep 60
    done
    P=$AREA/src/RecoTICL/Inference/data/showerShapePID
    for l in hi fin; do
      cp $D/models/mix/mixPID_${l}_${TAG}_mapped.onnx $P/.mixPID_${l}_$TAG.onnx.tmp && mv $P/.mixPID_${l}_$TAG.onnx.tmp $P/mixPID_${l}_$TAG.onnx || exit 1
    done
    ( cd $P && md5sum mixPID_hi_$TAG.onnx mixPID_fin_$TAG.onnx )
    OUTROOT=$D/dumps BLOCK=$W/blocks/m6s_$POINT.py exec bash $N/tools/pool3.sh $D/queue_${TAG}_dumps.txt "$@" ;;
  *) echo "STEP must be probe or dumps"; exit 2 ;;
esac
