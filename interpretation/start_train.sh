#!/bin/bash
# In felice-1gpu-nvl inside cmssw-el8 --nv: waits until every dump chain of TAG is done (count of rc=0 lines in the pool
# log; deadline 30 h; stops on more than 10 failed), writes the chain list and runs train_m6.sh PART=hyp and PART=share
# in parallel. usage: TAG=rt3p2 start_train.sh
W=/cms-hlt-nfs/user/fpantale/ticlv6tune/rt3
D=$W/${TAG:?TAG}
Q=$D/queue_${TAG}_dumps.txt
L=$Q.log
need=$(grep -c . $Q)
end=$(( $(date +%s) + 108000 ))
until [ "$(grep -c 'rc=0' $L 2>/dev/null)" -ge "$need" ]; do
  nf=$(grep -vc 'rc=0' $L 2>/dev/null)
  [ "${nf:-0}" -le 10 ] || { echo "$nf failed dumps" > $D/train/train.failed; exit 1; }
  [ "$(date +%s)" -lt "$end" ] || { echo "dumps not done after 30 h" > $D/train/train.failed; exit 1; }
  sleep 120
done
awk '/rc=0/ {print $3, $4}' $L | sort -u | while read -r s c; do echo "$s $D/dumps/$s/$(printf %06d $c)"; done > $D/train/chaindirs.txt
echo "$(date '+%F %T') chains $(wc -l < $D/train/chaindirs.txt)"
export TAG CHAINDIRS=$D/train/chaindirs.txt
PART=hyp bash $W/train/train_m6.sh > $D/train/hyp.log 2>&1 &
PART=share bash $W/train/train_m6.sh > $D/train/share.log 2>&1 &
wait
ls $D/train/*.done $D/train/*.failed 2>/dev/null
