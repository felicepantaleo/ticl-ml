# Stage-2 probe block (pidstudy/stage2/probe_block.py), appended to the reco template of a train chain: the hypothesis
# dump, the truth of every trackster per stage (StageTruthProbe) and the CLUE3DHigh and final tracksters (EDM output).
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
    masks=cms.VInputTag("hgcalMergeLayerClusters:InitialLayerClustersMask", "filteredLayerClustersCLUE3DHigh:CLUE3DHigh",
                        "ticlTrackstersCLUE3DHigh", "filteredLayerClustersRecovery:Recovery", "ticlTrackstersRecovery"),
    tracksters=cms.VInputTag("ticlTrackstersCLUE3DHigh", "ticlTrackstersRecovery", "ticlTracksterLinks",
                             "ticlTracksterLinksSuperclusteringDNN", "ticlTracksterInterpretations"),
    candidates=cms.InputTag("ticlCandidate"),
    graph=cms.InputTag("truthLogicalGraphProducer"),
    hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
    legs=cms.InputTag("truthBranchTargets", "truthToRecoTargetsReconstructableFinalState"),
    trackRecoToTruth=cms.InputTag("allTrackToTruthBranchAssociators", "generalTracksRecoToTruthFixed"),
    lcAllAbove=cms.double(1.0), lcPrescale=cms.uint32(20), minObjDeposit=cms.double(0.01))
process.truthBranchValidationPath += process.stageProbe
process.pidOut = cms.OutputModule("PoolOutputModule", fileName=cms.untracked.string("tracksters.root"),
    outputCommands=cms.untracked.vstring("drop *", "keep ticlTracksters_ticlTrackstersCLUE3DHigh__*",
                                         "keep ticlTracksters_ticlTracksterInterpretations__*"))
process.pidOutPath = cms.EndPath(process.pidOut)
process.schedule.append(process.pidOutPath)
