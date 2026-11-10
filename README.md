# ticl-ml

Preprocessing and training pipelines of the machine-learning models of TICLv6 (HGCAL reconstruction in CMSSW, Phase-2,
geometry D127, PU200).

Each pipeline has three parts:

1. A RECO job with a study dumper (a CMSSW `EDAnalyzer` in `cmssw/`) writes flat rows from simulated chains (50 events
   each).
2. Python scripts make the training tables from these rows (the preprocessing).
3. Python scripts train the model and export it to ONNX with the interface that the CMSSW producer reads.

## Models

The models are the defaults of TICLv6 on the branch `ticlv6-modifier` of felicepantaleo/cmssw (commit 6d6471d043b,
2026-11-08). They are in `models/`, with `models/md5sums.txt`.

| model | CMSSW path (`RecoTICL/...`) | md5 | pipeline | trained |
|---|---|---|---|---|
| mixture PID, CLUEstering tracksters | `Inference/data/showerShapePID/mixPID_hi_v2.onnx` | 49ea7f9a | `showerShapePID/` | 2026-11-06, run rt3p2 |
| mixture PID, final tracksters | `Inference/data/showerShapePID/mixPID_fin_v2.onnx` | 05edb7b7 | `showerShapePID/` | 2026-11-06, run rt3p2 |
| hypothesis arbitration MLP | `Interpretation/data/arbitration/hypothesis_mlp_v3.onnx` | 7558c581 | `interpretation/` | 2026-11-07, run rt3p2 |
| neutral species MLP | `Interpretation/data/arbitration/neutralSpecies_mlp_v3.onnx` | a27de914 | `interpretation/` | 2026-11-07, run rt3p2 |
| neutral energy share MLP | `Interpretation/data/neutralEnergy/neutralShare_mlp_v2.onnx` | d4e35188 | `interpretation/` | 2026-11-07, run rt3p2 |
| e/gamma supercluster score (xgboost) | `ParticleFlow/data/egammaSuperClusters/score_v2.onnx` | 195578a2 | `egammaSuperClusters/score_type/` | 2026-11-07, run rt3p2 |
| e/gamma supercluster type score (xgboost) | `ParticleFlow/data/egammaSuperClusters/type_v2.onnx` | 932ba57d | `egammaSuperClusters/score_type/` | 2026-11-07, run rt3p2 |
| e/gamma supercluster energy regression | `ParticleFlow/data/egammaSuperClusters/regression_v3.onnx` | 814834fc | `egammaSuperClusters/regression/` | 2026-10-22 |
| hadronic trackster calibration (quantiles) | `Interpretation/data/chargedEnergy/hadronicTracksterCalibration_v1.onnx` | b3daba33 | `chargedEnergy/` | 2026-10-14 |

Run rt3p2 is the retrain of 2026-11-06/07 on the tracksters of the tune point p2 (`common/harness/p2.py`). The md5 of
each output of that run is equal to the md5 of the deployed file.

Three outputs were made again from the stored intermediate files with the scripts of this repository. Each one is
byte-identical to the deployed file (checked 2026-11-10):

- `regression_v3.onnx`: `ens2onnx.py` from its 9 members.
- `hypothesis_mlp_v3.onnx`: `ens_export.py` from its 4 seeds.
- `mixPID_hi_v2.onnx`: `export_mix.py` from the trained network and the probe rows.

## Layout

| folder | content |
|---|---|
| `cmssw/` | study dumpers (CMSSW packages to copy into `src/`), the patch of the study-only hypothesis dump, the share model that gives 1 |
| `common/` | job and pool scripts of the dump production, the configuration blocks appended to the RECO job, the RECO and DIGI templates, the tune parameters |
| `showerShapePID/` | mixture PID: probe rows, flat tables, training, map onto the CNN scale |
| `interpretation/` | hypothesis, neutral species and neutral share models: tables, pairs, seeds, ensemble |
| `egammaSuperClusters/score_type/` | supercluster score and type score |
| `egammaSuperClusters/regression/` | supercluster energy regression |
| `chargedEnergy/` | hadronic trackster calibration |
| `models/` | the deployed ONNX files |

Every folder has a README with the inputs, the steps, the commands and the outputs.

## Samples

All samples are Phase-2 D127 at PU200 (era `Phase2C26I13M9`, pile-up library RelValMinBias_14TeV D127 with 900,000
events). A chain has 50 events. The MC-truth graph (`truth::Graph`) is built at DIGI and gives the labels.

| sample | kind | DIGI production | chains in run rt3p2 |
|---|---|---|---|
| TenTau, E 15-500 GeV, abs(eta) < 3.1 | `tt` | c1 (`ticlv6prod/tt`) | dumps 862, probes 32 |
| e/gamma guns | `eg` | c1 and c2 (`ticlv6prod/eg`, `ticlv6prod_c2/eg`) | dumps 236, probes 16, supercluster legs 68 |
| QCD | `qcd` | c2 (`ticlv6prod_c2/qcd`) | dumps 252, probes 16 |
| QCD, high pT-hat | `qcdhi` | c3 (`ticlv6prod_c3/qcdhi`) | dumps 64, probes 32 |

The supercluster legs also use 80 TenTau chains (1, 6, ..., 196 and 3, 8, ..., 198).

No training reads the validation chains of the TICLv6 A/B set: TenTau 2, 7, ..., 197, e/gamma 32-63, QCD 100-123.
The splits are by chain id, so all rows of one event are in one split. Each README gives the split of its model.

## Code and environment of run rt3p2

- RECO and dumps: CMSSW_20_1_X_2026-11-02-2300, el8_amd64_gcc14, local branch `rt3`. Its RecoTICL code is
  `ticlv6-modifier` at a6d5063e719 (the same code except `RecoTICL/Configuration/python/target.py` and one test),
  plus `cmssw/patches/0001-study-only-hypothesis-dump.patch`, plus `cmssw/CornettoStudy` and `cmssw/EGStudy`.
  CLUEstering has the fix of cms-patatrack/CLUEstering#644. Each job appends the p2 parameters
  (`common/harness/p2.py`). On `ticlv6-modifier` 6d6471d043b they are the defaults.
- RECO template: `common/cfg/reco_v6.py` (cmsDriver `RAW2DIGI,RECO`, `--procModifiers ticl_v6`).
- Training: the python of the same release (torch, onnx, onnxruntime, xgboost, onnxmltools, uproot, scikit-learn).
  The hypothesis seeds and the species model use a GPU (Tesla T4). The other models use the CPU.
- Machine: patatrack-bg-01. Data on the CMS NFS `/cms-hlt-nfs/user/fpantale`.

## Paths

The shell drivers (`*.sh`) are the scripts that ran, without changes, so they hold the paths of that run. The python
scripts find the other scripts of this repository relative to their own path. They read the data paths from
environment variables or arguments. To run a pipeline in another place, change the variables at the top of each
driver. This table gives the place in this repository of each path in the drivers:

| path in the drivers | in this repository |
|---|---|
| `$W` = `/cms-hlt-nfs/user/fpantale/ticlv6tune/rt3`: `*.sh` | `common/`, `showerShapePID/train_mix.sh`, `interpretation/{start_train,train_m6}.sh`, `egammaSuperClusters/score_type/*.sh` |
| `$W/lib/pidstudy` | `showerShapePID/pidstudy` |
| `$W/sc/scripts` | `egammaSuperClusters/score_type` |
| `$W/blocks`, `$W/harness` | `common/blocks`, `common/harness` |
| `$W/repo/ticlv6_baseline/bin/ticl_leg.sh` | `common/ticl_leg.sh` |
| `$N` = `/cms-hlt-nfs/user/fpantale/hackathon_retrain`: `tools/{pool3,job3}.sh`, `cfg/reco_v6.py` | `common/`, `common/cfg/reco_v6.py` |
| `$N/train/scripts` | `interpretation/hypothesis` |
| `$N/train/train_species_m6.py` | `interpretation/species` |
| `$N/train/{prepare,train}_share_m6.py` | `interpretation/share` |
| `$W/<TAG>/...` | outputs of a run (not in this repository) |

## Order of a full retrain

1. `common/setup_tag.sh`: blocks, queues and folders of one run.
2. Probe chains, then the mixture PIDs (`showerShapePID/`).
3. Dump chains with the new mixture PIDs, then the interpretation models (`interpretation/`).
4. Supercluster legs with all new models, then the score and type models (`egammaSuperClusters/score_type/`).
   `common/rest.sh` runs step 4, installs the models and runs the validation leg.

Each step waits for the end marker of the step before it, so all steps can start at once. See `common/README.md`.

Run rt3p2 did not retrain the energy regression and the hadronic calibration.
