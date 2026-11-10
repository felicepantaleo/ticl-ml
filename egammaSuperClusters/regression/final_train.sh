#!/bin/bash
# usage: final_train.sh <out dir> <cpus> <member spec> ...: trains each member on the whole training sample (PU200
# t200.npz weight 4, PU0 t0.npz, HE tH.npz of /dev/shm/fpantale/v3dev/sc3/train) on the Comp24 superclusters with
# 19 features, then merges the members into <out dir>/regression_v3.onnx (geometric mean).
# member spec: <name>:<mlp|xgb>:<weight>:<train3.py options with + for spaces>, e.g. m1:mlp:1:--seed+1
out=$1; cpus=$2; shift 2
T=/dev/shm/fpantale/v3dev/sc3/train; A=/home/fpantale/2026/vbf/v3dev/sc3/ana
mkdir -p $out; cd $out
files=()
for spec in "$@"; do
  IFS=: read -r name model w opts <<< "$spec"
  ext=onnx; [ "$model" = xgb ] && ext=json
  taskset -c $cpus python3 $A/train3.py $name.onnx ticlEGammaSuperClustersComp24 --nf 19 --model $model --threads 4 \
    --data ${T200:-$T/t200.npz}:4:all $T/t0.npz:1:all $T/tH.npz:1:all ${opts//+/ } > log_$name.txt 2>&1 &
  files+=($name.$ext:$w)
done
wait
taskset -c $cpus python3 $A/ens2onnx.py regression_v3.onnx "${files[@]}" 2>&1 | grep -v "^WARNING\|^I0000\|^E0000" | tail -1
md5sum regression_v3.onnx
