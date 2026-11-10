#!/bin/bash
# Installs the interpretation models of <TAG>/models/interp into the area $1 under the names of blocks/models_<TAG>.py.
# usage: TAG=rt3p2 install_models.sh <area>
W=/cms-hlt-nfs/user/fpantale/ticlv6tune/rt3
D=$W/${TAG:?TAG}
A=${1:?area}/src/RecoTICL/Interpretation/data
put() { cp "$1" "$2.tmp" && mv "$2.tmp" "$2" || exit 1; }
put $D/models/interp/hypothesis_mlp_m6.onnx $A/arbitration/hypothesis_mlp_$TAG.onnx
put $D/models/interp/neutralSpecies_mlp_m6.onnx $A/arbitration/neutralSpecies_mlp_$TAG.onnx
put $D/models/interp/neutralShare_mlp_m6.onnx $A/neutralEnergy/neutralShare_mlp_$TAG.onnx
md5sum $A/arbitration/*_$TAG.onnx $A/neutralEnergy/*_$TAG.onnx
