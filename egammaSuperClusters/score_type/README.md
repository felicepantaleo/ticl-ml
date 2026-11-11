# e/gamma supercluster score and type score

Models of `ticlEGammaSuperClusters` (`RecoTICL/ParticleFlow`):
- `score_v2.onnx` (`scoreModel`): the probability that a supercluster is a real e/gamma object.
- `type_v2.onnx` (`typeScoreModel`): the probability that the signal makes the supercluster, for the type of the
  photon candidates.

Both are xgboost models on the 35 score features (`scoreFeatures` of
`RecoTICL/ParticleFlow/interface/EGammaSuperClusterFeatures.h`), exported to ONNX (input "input" [N, 35], output
"score" [N]).

## Legs

148 RECO legs (`run_sc.sh`, queue `<TAG>/sc/queue_sc.txt`): e/gamma 0-31 and 64-99, TenTau 1, 6, ..., 196 and
3, 8, ..., 198. Each leg is `common/ticl_leg.sh <kind> <chain> sc_<TAG>` with the block `common/blocks/sc_leg_p2.py`
(template `common/blocks/sc_leg.py`):
- no score cut, no type score, no photon retype;
- all new models of the run (`common/blocks/models_rt3p2.py`);
- the probe `EGStudyTruthProbe` (`cmssw/EGStudy`, `customiseEGProbe.customise(process, v6=True)`) on e/gamma and
  TenTau files.

    TAG=rt3p2 VBF_WORK=<local dir> bash egammaSuperClusters/score_type/run_sc.sh "<cpus> <numa node>,..."

`common/rest.sh` runs this step after the interpretation models.

## Training (`train_sc.sh`)

    TAG=rt3p2 AREA=<cmssw area> CPUS=<cpu list> bash egammaSuperClusters/score_type/train_sc.sh

- `train_score.py <legs>/eg sc_<TAG> <out> --train-chains 0-31,64-89 --val-chains 90-99`.
  - Label 1: the signal (gun) deposit is at least 0.5 of the truth deposit of the supercluster.
  - Label 0: the signal deposit is below 0.5 and no gun particle is within dR 0.2 (a fake).
  - Other rows are not used.
  - xgboost depth 6, eta 0.05, at most 2000 rounds, with early stopping on the validation chains.
- `train_type.py <legs>/eg sc_<TAG> <legs>/tt sc_<TAG> <out>`.
  - Label 1: the signal deposit is at least 0.5 of the truth deposit. Label 0 otherwise (pile-up dominated).
  - Signal: the gun particles for e/gamma, the tau decay products for TenTau.
  - Training: e/gamma 0-31, 64-89 and TenTau 1, 6, ..., 196. Early stopping: e/gamma 90-99 and TenTau 3, 8, ..., 198.
- `ana_diag.py` holds `gunDR`, which `train_score.py` imports. It is also the diagnosis of the fakes.

## Run rt3p2

`logs/`: `score.log` (train 64351 rows in 58 chains, validation 11003 rows in 10 chains, best iteration 376,
validation logloss 0.0551), `type.log` (pile-up superclusters kept at a signal efficiency of 0.98 / 0.95 / 0.90:
0.263 / 0.122 / 0.049 on TenTau validation), `meta.json`.

## Score cuts at equal fake rate

A new score model has a different calibration, so the cuts of the old model do not give the same operating point.
`equal_fake_cuts.py` gives the cuts of the new model that keep the number of fake superclusters of a reference model at
its cuts, in raw Et 2-10 GeV and at or above 10 GeV, on the validation chains 90-99 of the supercluster legs. With the
reference score_v1 at 0.5 and 0.9, the rt3p2 score model gives 0.335 and 0.850 (68 and 7 fakes, as score_v1); the
defaults 0.35 and 0.85 come from the same scan with steps of 0.05. `common/rest.sh` runs it when `SC_REF` is set.

    python3 equal_fake_cuts.py <RUNDIR>/<TAG>/sc/legs/eg sc_<TAG> <reference onnx> <cut below 10 GeV> <cut at or above 10 GeV> <new onnx>
