# The TICLv6 models retrained on the p2 tracksters (rt3p2): mixture PIDs, hypothesis ensemble, neutral species, neutral share.
process.ticlTrackstersCLUEstering.pluginInferenceAlgoTracksterInferenceByShowerShape.onnxModelPath = 'RecoTICL/Inference/data/showerShapePID/mixPID_hi_rt3p2.onnx'
process.ticlTracksterInterpretations.pluginInferenceAlgoTracksterInferenceByShowerShape.onnxModelPath = 'RecoTICL/Inference/data/showerShapePID/mixPID_fin_rt3p2.onnx'
process.ticlTracksterInterpretations.arbitrationModelFile = 'RecoTICL/Interpretation/data/arbitration/hypothesis_mlp_rt3p2.onnx'
process.ticlTracksterInterpretations.neutralModelFile = 'RecoTICL/Interpretation/data/arbitration/neutralSpecies_mlp_rt3p2.onnx'
process.ticlCandidate.neutralShareModel = 'RecoTICL/Interpretation/data/neutralEnergy/neutralShare_mlp_rt3p2.onnx'
