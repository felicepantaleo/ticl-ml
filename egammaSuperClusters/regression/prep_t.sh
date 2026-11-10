#!/bin/bash
# usage: prep_t.sh: per-photon npz files of the training sample from the dumps of the training RECO jobs that ended
# with rc 0 (PU200 files 100-105 -> t200.npz, PU0 -> t0.npz, HE -> tH.npz), in /dev/shm/fpantale/v3dev/sc3/train.
D=/dev/shm/fpantale/v3dev/sc3; T=$D/train; A=/home/fpantale/2026/vbf/v3dev/sc3/ana
L=ticlEGammaSuperClustersNoComp,ticlEGammaSuperClustersW50,ticlEGammaSuperClustersSeed,ticlEGammaSuperClustersComp24
f=(); for k in 100 101 102 103 104 105; do grep -aq '^rc 0$' $D/reco_T_$k/rc 2>/dev/null && f+=($D/reco_T_$k/dump.root); done
echo "PU200 files: ${#f[@]}"
python3 $A/dump2npz.py $T/t200.npz $L "${f[@]}" 2>&1 | tail -1 &
grep -aq '^rc 0$' $D/reco0_T/rc && python3 $A/dump2npz.py $T/t0.npz $L $D/reco0_T/dump.root 2>&1 | tail -1 &
grep -aq '^rc 0$' $D/recoH_T/rc && python3 $A/dump2npz.py $T/tH.npz $L $D/recoH_T/dump.root 2>&1 | tail -1 &
wait
