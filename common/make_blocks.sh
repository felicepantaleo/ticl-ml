#!/bin/bash
# Writes the job blocks of one retrain: blocks/{probe,m6,m6s,sc_leg}_<POINT>.py, blocks/models_<TAG>.py and
# blocks/val_<POINT>rt.py, from blocks/templates and the parameter block harness/<POINT>.py.
# usage: POINT=p2 TAG=rt3p2 make_blocks.sh
W=/cms-hlt-nfs/user/fpantale/ticlv6tune/rt3
P=${POINT:?POINT} T=${TAG:?TAG}
B=$W/blocks
[ -s $W/harness/$P.py ] || { echo "no $W/harness/$P.py"; exit 1; }
params() {
  sed -e "s|^sys.path.insert(0, .*)|sys.path.insert(0, '$W/harness')|" $W/harness/$P.py
}
# Template text with the names of this retrain
names() {
  sed -E -e "s/mixPID_(hi|fin)_rt7/mixPID_\1_$T/g" -e "s@ticlv6tune/rt7/blocks/models_rt7.py@ticlv6tune/rt3/blocks/models_$T.py@" |
    sed -e "s|on the t7 tracksters|on the $P tracksters|" -e "s|models for t7 (rt7)|models for $P ($T)|" \
        -e "s|the rt7 models|the $T models|" -e "s|the t7 parameters last|the $P parameters last|"
}
for b in probe m6 sc_leg; do
  { names < $B/templates/$b.py; params; } > $B/${b}_$P.py.tmp && mv $B/${b}_$P.py.tmp $B/${b}_$P.py
done
# m6s: the template without its last 6 lines, the parameters, then the 6 share-row lines
{ head -n -6 $B/templates/m6s.py | names; params; echo; tail -n 6 $B/templates/m6s.py; } > $B/m6s_$P.py.tmp &&
  mv $B/m6s_$P.py.tmp $B/m6s_$P.py
cat > $B/models_$T.py <<PY
# The TICLv6 models retrained on the $P tracksters ($T): mixture PIDs, hypothesis ensemble, neutral species, neutral share.
process.ticlTrackstersCLUEstering.pluginInferenceAlgoTracksterInferenceByShowerShape.onnxModelPath = 'RecoTICL/Inference/data/showerShapePID/mixPID_hi_$T.onnx'
process.ticlTracksterInterpretations.pluginInferenceAlgoTracksterInferenceByShowerShape.onnxModelPath = 'RecoTICL/Inference/data/showerShapePID/mixPID_fin_$T.onnx'
process.ticlTracksterInterpretations.arbitrationModelFile = 'RecoTICL/Interpretation/data/arbitration/hypothesis_mlp_$T.onnx'
process.ticlTracksterInterpretations.neutralModelFile = 'RecoTICL/Interpretation/data/arbitration/neutralSpecies_mlp_$T.onnx'
process.ticlCandidate.neutralShareModel = 'RecoTICL/Interpretation/data/neutralEnergy/neutralShare_mlp_$T.onnx'
PY
cat > $B/val_${P}rt.py <<PY
# Validation leg ${P}rt: the $P parameters with the models retrained on the $P tracksters (interpretation, PID, supercluster).
exec(open("$W/harness/$P.py").read())
exec(open("$B/models_$T.py").read())
exec(open("$B/models_sc_$T.py").read())
PY
ls -la $B/*_$P.py $B/*_$T.py $B/val_${P}rt.py
