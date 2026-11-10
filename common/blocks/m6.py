# M6d dump block (copy of the block of ticlv6retrain/m6d/run_m6.sh): hypothesis dump and share-model rows.
# Hypothesis dump (study build only); added when the template does not hold it.
process.ticlTracksterInterpretations.dumpHypotheses = True
if not hasattr(process, "hypDumper"):
    process.hypDumper = cms.EDAnalyzer("HypothesisTruthDumper",
        hypothesisModule=cms.string("ticlTracksterInterpretations"),
        layerClusters=cms.InputTag("hgcalMergeLayerClusters"),
        tracks=cms.InputTag("generalTracks"),
        trackRecoToTruth=cms.InputTag("allTrackToTruthBranchAssociators", "generalTracksRecoToTruthFixed"),
        graph=cms.InputTag("truthLogicalGraphProducer"),
        hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
        legs=cms.InputTag("truthBranchTargets", "truthToRecoTargetsReconstructableFinalState"),
        maxSharedFraction=cms.double(0.2))
    process.truthBranchValidationPath += process.hypDumper
# Share-model rows: no neutral energy correction and no minimum energy on the final candidates, and the energy probe.
process.ticlCandidate.neutralShareModel = "RecoTICL/Interpretation/data/neutralEnergy/neutralShare_ones_study.onnx"
process.ticlCandidate.neutralMinEnergy = 0.
process.truthBranchTargetsNoFloor = process.truthBranchTargets.clone()
process.truthBranchTargetsNoFloor.branchSelector.ptMin = 0.0
process.truthBranchTracksterAssociatorsNoFloor = process.truthBranchTracksterAssociators.clone(
    recoCollections=["ticlTracksterInterpretations"],
    targetsSrc=("truthBranchTargetsNoFloor", "selectedRoots"),
    assignableTargetsSrc=("truthBranchTargetsNoFloor", "assignableRoots"))
process.energyProbe = cms.EDAnalyzer("CandidateEnergyProbe",
    candidates=cms.InputTag("ticlCandidate"), tracks=cms.InputTag("generalTracks"),
    trackMode=cms.InputTag("ticlTracksterInterpretations", "trackMode"),
    trackRecoToTruth=cms.InputTag("allTrackToTruthBranchAssociators", "generalTracksRecoToTruthFixed"),
    graph=cms.InputTag("truthLogicalGraphProducer"), hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
    tracksterRecoToTruth=cms.InputTag("truthBranchTracksterAssociatorsNoFloor", "ticlTracksterInterpretationsRecoToTruthAdaptiveNominal"),
    layerClusters=cms.InputTag("hgcalMergeLayerClusters"), tracksters=cms.InputTag("ticlTracksterInterpretations"),
    minAbsEta=cms.double(1.5), maxAbsEta=cms.double(3.0))
process.truthBranchValidationPath += process.truthBranchTargetsNoFloor + process.truthBranchTracksterAssociatorsNoFloor + process.energyProbe
# The mixture PIDs retrained on the t7 tracksters: CLUEstering tracksters (hi) and final tracksters (fin).
process.ticlTrackstersCLUEstering.pluginInferenceAlgoTracksterInferenceByShowerShape.onnxModelPath = 'RecoTICL/Inference/data/showerShapePID/mixPID_hi_rt7.onnx'
process.ticlTracksterInterpretations.pluginInferenceAlgoTracksterInferenceByShowerShape.onnxModelPath = 'RecoTICL/Inference/data/showerShapePID/mixPID_fin_rt7.onnx'

