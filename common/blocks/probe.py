# Mixture-PID probe block of the CLUEstering retrain (copy of pidstudy/mix/probe_block.py for the CLUEstering tracksters), appended to the reco template of a train chain: the hypothesis
# dump, the truth of every trackster per stage (StageTruthProbe) and the CLUEstering and final tracksters (EDM output).
process.options.wantSummary = True
if hasattr(process, "jetsOutPath"):
    process.schedule.remove(process.jetsOutPath)
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
process.stageProbe = cms.EDAnalyzer("StageTruthProbe",
    layerClusters=cms.InputTag("hgcalMergeLayerClusters"),
    masks=cms.VInputTag("hgcalMergeLayerClusters:InitialLayerClustersMask", "hgcalMergeLayerClusters:InitialLayerClustersMask",
                        "ticlTrackstersCLUEstering", "filteredLayerClustersRecovery:Recovery", "ticlTrackstersRecovery"),
    tracksters=cms.VInputTag("ticlTrackstersCLUEstering", "ticlTrackstersRecovery", "ticlTracksterLinks",
                             "ticlTracksterLinksSuperclusteringDNN", "ticlTracksterInterpretations"),
    candidates=cms.InputTag("ticlCandidate"),
    graph=cms.InputTag("truthLogicalGraphProducer"),
    hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
    legs=cms.InputTag("truthBranchTargets", "truthToRecoTargetsReconstructableFinalState"),
    trackRecoToTruth=cms.InputTag("allTrackToTruthBranchAssociators", "generalTracksRecoToTruthFixed"),
    lcAllAbove=cms.double(1.0), lcPrescale=cms.uint32(20), minObjDeposit=cms.double(0.01))
process.truthBranchValidationPath += process.stageProbe
process.pidOut = cms.OutputModule("PoolOutputModule", fileName=cms.untracked.string("tracksters.root"),
    outputCommands=cms.untracked.vstring("drop *", "keep ticlTracksters_ticlTrackstersCLUEstering__*",
                                         "keep ticlTracksters_ticlTracksterInterpretations__*"))
process.pidOutPath = cms.EndPath(process.pidOut)
process.schedule.append(process.pidOutPath)
# The CNN PIDs at both levels (the scale onto which the new PID models are mapped); the interpretation models are the
# defaults of the area (M6d in run rt3p2).
process.ticlTrackstersCLUEstering.inferenceAlgo = 'TracksterInferenceByCNN'
process.ticlTracksterInterpretations.pidInferenceAlgo = ''
process.ticlTracksterInterpretations.pluginInferenceAlgoTracksterInferenceByPFN.doPID = 1
process.ticlTracksterInterpretations.pluginInferenceAlgoTracksterInferenceByPFN.onnxPIDModelPath = 'RecoHGCal/TICL/data/ticlv5/onnx_models/CNN/linking/id_v0.onnx'

