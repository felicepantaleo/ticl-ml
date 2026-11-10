#!/bin/bash
# Prepares one retrain of the fixed code (area rt3) for one tuned point: the blocks, the work folder <TAG> and its
# queues. The queues are the rt7 queues (H9 files without the validation files tt 2..197 step 5, eg 32-63, qcd 100-123).
# usage: POINT=p2 TAG=rt3p2 setup_tag.sh
W=/cms-hlt-nfs/user/fpantale/ticlv6tune/rt3
O=/cms-hlt-nfs/user/fpantale/ticlv6tune/rt7
P=${POINT:?POINT} T=${TAG:?TAG}
D=$W/$T
POINT=$P TAG=$T bash $W/make_blocks.sh > /dev/null || exit 1
mkdir -p $D/probe $D/dumps $D/models/mix $D/models/interp $D/models/sc $D/train $D/sc/legs
# Probe: the rt7 probe queue without e/gamma 40-55 (validation files) and with e/gamma 100-115 of the second production.
{ awk '!($2 == "eg" && $3 >= 32 && $3 <= 63)' $O/queue_rt7_probe.txt; awk '{print}' $O/queue_rt7_probe2.txt; } > $D/queue_${T}_probe.txt
# Dumps: the 1414 rt7 dump chains (TenTau, e/gamma, QCD, high-pT QCD), one pass with the m6s block.
cp $O/queue_rt7_m6.txt $D/queue_${T}_dumps.txt
# Supercluster legs: the rt7 list (e/gamma 0-31, 64-99; TenTau 1,6..196 and 3,8..198) with the leg sc_<TAG>.
awk -v l=sc_$T '{print $1, $2, l}' $O/sc/queue_sc7.txt > $D/sc/queue_sc.txt
A=/home/fpantale/2026/ticlv6/hackathon/vtx/rt3/CMSSW_20_1_X_2026-11-02-2300
echo "sc_$T $A reco_v6.py $W/blocks/sc_leg_$P.py" > $D/sc/legs.conf
echo "POINT=$P TAG=$T" > $D/point.env
wc -l $D/queue_${T}_probe.txt $D/queue_${T}_dumps.txt $D/sc/queue_sc.txt | head -3
