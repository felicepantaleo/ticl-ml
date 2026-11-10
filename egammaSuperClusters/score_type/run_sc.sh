#!/bin/bash
# Training legs of the TICL e/gamma supercluster models of TAG (<TAG>/sc/queue_sc.txt: e/gamma 0-31, 64-99; TenTau
# 1,6..196 and 3,8..198; never the validation chains). Each line runs repo/ticlv6_baseline/bin/ticl_leg.sh with the leg
# sc_<TAG> of <TAG>/sc/legs.conf. Claims by mkdir in <TAG>/sc/claims, one rc line per leg in rc.txt; stops claiming after
# 3 failed legs. Outputs <TAG>/sc/legs/<kind>/sc_<TAG>/<chain>; marker <TAG>/sc/ALL_DONE when every line has an rc.
# usage: TAG=rt3p2 VBF_WORK=<local work dir> run_sc.sh "<cpus> <membind>,..."
W=/cms-hlt-nfs/user/fpantale/ticlv6tune/rt3
S=$W/${TAG:?TAG}/sc
J=$W/repo/ticlv6_baseline/bin/ticl_leg.sh
Q=$S/queue_sc.txt
export VBF_BASE=$S VBF_OUT=$S/legs VBF_THREADS=${VBF_THREADS:-16} CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=
: "${VBF_WORK:?VBF_WORK}"
mkdir -p $S/claims $S/logs
touch $S/rc.txt
worker() {
  local i=0
  while read -r k c l; do
    i=$((i + 1))
    [ "$(grep -vc 'rc=0' $S/rc.txt)" -ge 3 ] && break
    mkdir $S/claims/$i 2>/dev/null || continue
    numactl --physcpubind=$1 --membind=$2 bash $J $k $c $l > $S/logs/${k}_${c}_$l.log 2>&1
    echo "$k $c $l rc=$?" >> $S/rc.txt
  done < $Q
}
IFS=, read -ra slots <<< "$1"
for s in "${slots[@]}"; do set -- $s; worker $1 $2 & sleep 10; done
wait
[ "$(wc -l < $S/rc.txt)" -ge "$(wc -l < $Q)" ] && echo "$(grep -c 'rc=0' $S/rc.txt) ok $(grep -vc 'rc=0' $S/rc.txt) failed" > $S/ALL_DONE
