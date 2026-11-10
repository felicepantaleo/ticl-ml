# CMSSW study code of the dumps

These are study packages, not for integration in CMSSW. Copy a package into `<area>/src` and run `scram b` (on one
socket).

| package | plugin | used by | built in |
|---|---|---|---|
| `CornettoStudy/Probe` | `StageTruthProbe` | `showerShapePID` (probe chains) | rt3 area (CMSSW_20_1_X_2026-11-02-2300) |
| `CornettoStudy/Probe` | `HypothesisTruthDumper` | `interpretation` (hypothesis and species rows), also in the probe chains | rt3 area |
| `CornettoStudy/Probe` | `CandidateEnergyProbe` | `interpretation` (share rows) | rt3 area |
| `CornettoStudy/Probe` | `TICLCalibrationProbe` | `chargedEnergy` | p4dev area (CMSSW_20_1_0_pre4, 2026-10-14) |
| `EGStudy/Probe` | `EGStudyTruthProbe`, `customiseEGProbe.py` | `egammaSuperClusters/score_type` | rt3 area |
| `SC3Study/Dump` | `EGammaSCFeatureDumper` | `egammaSuperClusters/regression` | sc3 area (CMSSW_20_1_0_pre4, branch photon-sc3) |

`CornettoStudy/Probe/plugins/BuildFile.xml` is the file of the rt3 area. That package has more probes, which are not
in this repository. The BuildFile builds every `.cc` of the folder.

The probes read the MC-truth graph (`truthLogicalGraphProducer`, `truthLogicalGraphHitIndexProducer`) and the truth
targets and associators of the truth validation (`truthBranchTargets`, `allTrackToTruthBranchAssociators`,
`truthBranchTracksterAssociators`). So the RECO job needs the truth validation path (`truthBranchValidationPath`) and a
DIGI file with the truth graph.

## patches

- `0001-study-only-hypothesis-dump.patch`: the parameter `dumpHypotheses` of `TICLInterpretationProducer`. With
  `dumpHypotheses = True` the producer puts flat products of the hypotheses: model features, weights and acceptance,
  the claimed layer clusters and the photon-winner flag of each neutral. `HypothesisTruthDumper` reads them. The
  commit is 1886eaabc2f of the rt3 area, on top of the RecoTICL code of `ticlv6-modifier` (see the top README).
- `neutralShare_ones_study.onnx`: a share model with output 1. The share dumps use it, so the candidates keep the full
  trackster energy. Install it as `RecoTICL/Interpretation/data/neutralEnergy/neutralShare_ones_study.onnx`.
