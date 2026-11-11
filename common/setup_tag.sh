#!/bin/bash
# Prepares one retrain for one tuned point: the blocks, the work folder <TAG> and its
# queues (common/queues: H9 files without the validation files tt 2..197 step 5, eg 32-63, qcd 100-123). AREA is the
# CMSSW area of the supercluster legs.
# usage: RUNDIR=<run folder> AREA=<cmssw area> POINT=p2 TAG=rt3p2 setup_tag.sh
X=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
W=${RUNDIR:?RUNDIR}
P=${POINT:?POINT} T=${TAG:?TAG}
D=$W/$T
POINT=$P TAG=$T bash $X/common/make_blocks.sh > /dev/null || exit 1
mkdir -p $D/probe $D/dumps $D/models/mix $D/models/interp $D/models/sc $D/train $D/sc/legs
# Probe: 96 chains (TenTau, e/gamma 100-115, QCD, high-pT QCD).
cp $X/common/queues/probe.txt $D/queue_${T}_probe.txt
# Dumps: 1414 chains (TenTau, e/gamma, QCD, high-pT QCD), one pass with the m6s block.
cp $X/common/queues/dumps.txt $D/queue_${T}_dumps.txt
# Supercluster legs: e/gamma 0-31, 64-99; TenTau 1,6..196 and 3,8..198, with the leg sc_<TAG>.
awk -v l=sc_$T '{print $1, $2, l}' $X/common/queues/sc.txt > $D/sc/queue_sc.txt
echo "sc_$T ${AREA:?AREA} reco_v6.py $W/blocks/sc_leg_$P.py" > $D/sc/legs.conf
echo "POINT=$P TAG=$T" > $D/point.env
wc -l $D/queue_${T}_probe.txt $D/queue_${T}_dumps.txt $D/sc/queue_sc.txt | head -3
