#!/bin/bash
# Trains the supercluster score (train_score.py, e/gamma 0-31,64-89, early stopping 90-99) and type score (train_type.py,
# e/gamma + TenTau, see its header) of TAG on the sc_<TAG> legs, in the CMSSW environment of AREA, on CPUS.
# Outputs <TAG>/models/sc/score and <TAG>/models/sc/type (score.onnx each); marker <TAG>/models/sc/train.done or train.failed.
# usage: RUNDIR=<run folder> TAG=rt3p2 AREA=<cmssw area> CPUS=<cpu list> train_sc.sh
X=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
W=${RUNDIR:?RUNDIR}
D=$W/${TAG:?TAG}
S=$D/sc
O=$D/models/sc
fail() { echo "$1" > $O/train.failed; exit 1; }
mkdir -p $O; rm -f $O/train.done $O/train.failed
[ -f $S/ALL_DONE ] || fail "legs not done"
cd "${AREA:?AREA}/src" && source /cvmfs/cms.cern.ch/cmsset_default.sh && eval "$(SCRAM_IGNORE_HOOKS=1 scram runtime -sh)" || fail cmsenv
taskset -c "${CPUS:?CPUS}" python3 -u $X/egammaSuperClusters/score_type/train_score.py $S/legs/eg sc_$TAG $O/score --train-chains 0-31,64-89 \
  --val-chains 90-99 > $O/score.log 2>&1 || fail score
taskset -c "$CPUS" python3 -u $X/egammaSuperClusters/score_type/train_type.py $S/legs/eg sc_$TAG $S/legs/tt sc_$TAG $O/type > $O/type.log 2>&1 || fail type
md5sum $O/score/score.onnx $O/type/score.onnx > $O/md5sums.txt
echo "$(date '+%F %T') done" > $O/train.done
