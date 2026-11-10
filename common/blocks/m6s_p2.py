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
# The mixture PIDs retrained on the p2 tracksters: CLUEstering tracksters (hi) and final tracksters (fin).
process.ticlTrackstersCLUEstering.pluginInferenceAlgoTracksterInferenceByShowerShape.onnxModelPath = 'RecoTICL/Inference/data/showerShapePID/mixPID_hi_rt3p2.onnx'
process.ticlTracksterInterpretations.pluginInferenceAlgoTracksterInferenceByShowerShape.onnxModelPath = 'RecoTICL/Inference/data/showerShapePID/mixPID_fin_rt3p2.onnx'

# Tuned parameters p2 (tune3 it000s2): set 0 of /data/fpantale/hackathon_vtx/tune/tune3/test_points.json
import sys
sys.path.insert(0, '/cms-hlt-nfs/user/fpantale/ticlv6tune/rt3/harness')
from make_tune_cfg import setParameter
setParameter(process.hgcalCLUEsteringEE, ['deltac'], [1.3842619336199442, 'float'])
setParameter(process.hgcalCLUEsteringEE, ['kappa'], [7.70371499020564, 'float'])
setParameter(process.hgcalCLUEsteringEE, ['outlierDeltaFactor'], [1.193126585361772, 'float'])
setParameter(process.hgcalCLUEsteringHSi, ['deltac'], [1.2857984208454247, 'float'])
setParameter(process.hgcalCLUEsteringHSi, ['kappa'], [5.487172898811695, 'float'])
setParameter(process.hgcalCLUEsteringHSi, ['outlierDeltaFactor'], [2.3419098337644857, 'float'])
setParameter(process.hgcalCLUEsteringHSci, ['deltac'], [0.03388934280881225, 'float'])
setParameter(process.hgcalCLUEsteringHSci, ['kappa'], [9.710221064795272, 'float'])
setParameter(process.hgcalCLUEsteringHSci, ['outlierDeltaFactor'], [2.5685114022807243, 'float'])
setParameter(process.ticlCLUEstering, ['deltac'], [1.6997148722485347, 'float'])
setParameter(process.ticlCLUEstering, ['rhoc'], [0.30483237976950317, 'float'])
setParameter(process.ticlCLUEstering, ['outlierDistance'], [3.3167447136046313, 'float'])
setParameter(process.ticlCLUEstering, ['seedingDistance'], [2.7718042783093337, 'float'])
setParameter(process.ticlCLUEstering, ['layerScale'], [3.377194342176505, 'float'])
setParameter(process.ticlCLUEstering, ['rhocEtaExponent'], [3.0117299351406634, 'float'])
setParameter(process.ticlCLUEstering, ['rhocPivotRadius'], [0.39770189059528605, 'float'])
setParameter(process.ticlCLUEstering, ['minLayerClusters'], [2, 'uint'])
setParameter(process.ticlCLUEstering, ['minSiliconCells'], [2, 'uint'])
setParameter(process.ticlTracksterLinks, ['linkingPSet', 'maxLongitudinalDistance'], [96.70198965421127, 'float'])
setParameter(process.ticlTracksterLinks, ['linkingPSet', 'maxBackwardDistance'], [13.713195659245459, 'float'])
setParameter(process.ticlTracksterLinks, ['linkingPSet', 'emTransverseRadius0'], [1.7134312957413216, 'float'])
setParameter(process.ticlTracksterLinks, ['linkingPSet', 'emTransverseSlope'], [0.012746363098379409, 'float'])
setParameter(process.ticlTracksterLinks, ['linkingPSet', 'transverseRadius0'], [2.5920967073489423, 'float'])
setParameter(process.ticlTracksterLinks, ['linkingPSet', 'transverseSlope'], [0.01787202233663772, 'float'])
setParameter(process.ticlTracksterLinks, ['linkingPSet', 'timeCompatibilityNSigma'], [2.4490096877491254, 'float'])
setParameter(process.ticlTracksterLinks, ['linkingPSet', 'typeVetoProbability'], [0.5576434880810307, 'float'])
setParameter(process.ticlTracksterLinks, ['linkingPSet', 'minEmittedEnergy'], [1.6552707168382577, 'float'])
setParameter(process.ticlTracksterLinks, ['linkingPSet', 'minEmittedPt'], [0.9236337717849741, 'float'])
setParameter(process.ticlTracksterLinks, ['linkingPSet', 'minEmittedEnergyEM'], [0.8516300438820504, 'float'])
setParameter(process.ticlTracksterLinks, ['linkingPSet', 'minEmittedPtEM'], [0.5, 'float'])
setParameter(process.ticlTracksterLinks, ['linkingPSet', 'mergeFragments'], [False, 'bool'])
setParameter(process.ticlTracksterLinks, ['linkingPSet', 'mergeDropped'], [False, 'bool'])

# Share rows: the candidates keep the interpretation tracksters, which the neutral share model reads at run time.
process.ticlCandidate.candidateTracksters.assignment = "none"
# Without per-candidate tracksters, ticlCandidate holds no trackster collection for the trackster associator and validator.
for _m in (process.truthBranchTracksterAssociators, process.truthBranchTracksterValidator):
    _m.recoCollections = type(_m.recoCollections)(*[x for x in _m.recoCollections
                                                     if (x if isinstance(x, str) else x.getModuleLabel()) != "ticlCandidate"])
