# e/gamma supercluster energy regression

Model `regression_v3.onnx` of `ticlEGammaSuperClusters` (`regressionModel`, with `regressionInputs = 19` and
`regressionMaxRatio = 1.5`). Input "input" [N, 19], the first 19 features of
`RecoTICL/ParticleFlow/interface/EGammaSuperClusterFeatures.h`, with the CE-E cell-level shapes. Output "ratio" [N] =
E_gen / E_raw (the median).

The superclusters use the composition `Comp24`: a time window of 0.05 ns for abs(eta) < 2.4, and the seed trackster
only for abs(eta) >= 2.4.

## Samples

Photon guns of the study "sc3" (2026-10-22), with their own GEN-SIM and DIGI. None of them is a validation file:
- PU200: files 100, 101, 102, 104 (500 events each, E 20-500 GeV).
- PU0: 3000 events.
- High energy (HE): 500 events, E 400-1200 GeV.

## Steps

1. RECO with the dumper (`mk_reco3.py` writes the job configuration).
   - The job has copies of `ticlEGammaSuperClusters` with the compositions `NoComp`, `W50`, `Seed`, `Comp24`.
   - `EGammaSCFeatureDumper` (`cmssw/SC3Study`) writes raw energy, corrected energy and the 19 features of each
     supercluster to `dump.root`.
2. `prep_t.sh` runs `dump2npz.py`. For each generator photon in 1.6 < abs(eta) < 2.8, it takes the nearest
   supercluster (dR < 0.1). Outputs `t200.npz` (PU200), `t0.npz` (PU0) and `tH.npz` (HE).
3. `final_train.sh <out dir> <cpus> <member spec> ...` trains each member with `train3.py`. Spec
   `<name>:<mlp|xgb>:<weight>:<options>`, with `+` for a space in the options. The target is log(E_gen / E_raw) with
   an L1 loss (the median), on the `Comp24` superclusters. PU200 has weight 4. Then `ens2onnx.py` merges the members
   into one ONNX, whose output is the weighted geometric mean of the members.
4. `check_parity.py` compares the CMSSW energy with the offline model on the dumped features.
   `compare3_predict.py` holds the model evaluation.

## Members of regression_v3

| name | model | weight |
|---|---|---|
| m1-m4 | MLP 64x64, seeds 1-4 | 1 each |
| L1, L2 | MLP 128x128x128, seeds 1-2 | 1 each |
| x1 | xgboost, depth 6, pseudo-Huber (685 trees after early stopping) | 1 |
| s1, s2 | scale-only MLP on the columns 0, 1, 14, seeds 1-2 | 3.5 each |

On 2026-11-10 `ens2onnx.py` of this repository merged the 9 stored members with these weights. The output is
byte-identical to the deployed file (md5 814834fc):

    ens2onnx.py regression_v3.onnx m1.onnx:1 m2.onnx:1 m3.onnx:1 m4.onnx:1 L1.onnx:1 L2.onnx:1 x1.json:1 s1.onnx:3.5 s2.onnx:3.5

The member options of `train3.py` come from the study notes: `--hidden 128,128,128` for L1 and L2, `--model xgb` for
x1, `--cols 0,1,14` for s1 and s2. No log of the exact `final_train.sh` command was kept.

The study ran in CMSSW_20_1_0_pre4 (branch `photon-sc3`, 0ebd38ad4fe, of the VBF study).
