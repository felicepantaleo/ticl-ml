# Hadronic trackster calibration

Model `hadronicTracksterCalibration_v1.onnx` of the TICLv6 candidate arbitration
(`RecoTICL/Interpretation/plugins/TICLCandidateArbitrationProducer.cc`). Charged hadrons share some hadronic
tracksters. For such a trackster, the network predicts the 16, 50 and 84% quantiles of
t = (sum of the true energies of the particles of the tracks) / (regressed energy). It uses calorimeter features
only, because the track momentum is the other measurement in the average. The producer fits each track energy with
inverse-variance weights: the calibrated trackster energy (q50) with the error (q84 - q16) / 2, and the track.

- Input "input" [N, 10]: log regE, log rawE, emFrac, abs(eta), log1p nLC, sigPCA0, sigPCA1, sigPCA2, nTracks,
  nJetMembers.
- Network: standardised inputs, MLP 64-64 SiLU, 3 outputs. The outputs are ordered log-quantiles, and the model gives
  exp of them ("quantiles" [N, 3]).
- Loss: pinball loss. Adam 2e-3.
- Rows: the `calibProbe/groups` with a hadronic trackster, where every track matches a truth particle with
  purity >= 0.5.
- Split: by chain, 25% of the chains (random, fixed seed) are validation.

## Steps

1. Dumps (`run_cal.sh <tt|eg|qcd> <chain> <cpus>`, pool `pool_cal.sh`): RECO v6 of the DIGI of the chain with
   `TICLCalibrationProbe` (`cmssw/CornettoStudy`) and the adaptive trackster associator without the pT floor. The
   code was the p4dev area (CMSSW_20_1_0_pre4, branch `p4/ticlv6-shape-pid`, 2026-10-14). Chains: TenTau 200-248
   (40), QCD 24-55 (32). The e/gamma chains were for a regression that is not deployed.
2. Training: `train_hadcal.py <out.onnx>` reads `calib/{tt,qcd}/<chain>/ntuple.root` (`C` at the top of the script).

## Run of 2026-10-14

`logs/hadcal_v1.log`:
- 321181 groups: train 240396, validation 80785 (72 chains, 18 of them validation);
- 198 epochs, validation pinball loss 0.2432;
- coverage of [q16, q84] 0.677 (0.68 expected), median of t/q50 0.998;
- ONNX parity 1.1e-5.
