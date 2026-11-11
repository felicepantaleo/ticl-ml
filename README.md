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

With `--procModifiers ticl_v6,ticl_v6_heterogeneous`, two models run on the device with PyTorch (TorchScript files, branch `hgcalOnly-workflow` 5a3ff681d99). They are exports of ONNX models, with no new training (`heterogeneous/`):

| model | CMSSW path (`RecoTICL/...`) | md5 | export of |
|---|---|---|---|
| device mixture PID, CLUEstering tracksters | `Inference/data/showerShapePID/mixPID_hi_v2.pt` | 14aac6c2 | `mixPID_hi_v2.onnx` |
| device superclustering DNN | `Superclustering/data/supercls_v3.pt` | 04ee12c5 | `RecoHGCal/TICL/data/superclustering/supercls_v3.onnx` (TICLv5 model, cms-data) |

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
| `heterogeneous/` | TorchScript models of the device inference, the export script, the parity check |
| `models/` | the deployed ONNX files |

Every folder has a README with the inputs, the steps, the commands and the outputs.

## Samples

All samples are Phase-2 D127 (era `Phase2C26I13M9`, geometry `ExtendedRun4D127`, conditions
`auto:phase2_realistic_T35`, beam spot `DBrealisticHLLHC`) at PU200. GEN-SIM runs with `--procModifiers enableTruth`.
The MC-truth graph (`truth::Graph`) is built at DIGI and gives the labels. A chain has 50 events.

### Samples and runTheMatrix workflows

The workflow is the runTheMatrix upgrade workflow (`runTheMatrix.py -w upgrade`, CMSSW_20_1_X) with the same generator
fragment. The `.88` variant has `enableTruth` at GEN-SIM, as these samples. The chains are a private production of the
same configuration, with the differences in the last column.

| sample | kind | GEN fragment | workflow | differences from the workflow |
|---|---|---|---|---|
| TenTau | `tt` | `TenTau_E15to500_Eta3p1_pythia8_cfi` | 37887.88 (PU200), 37687.88 (no PU) | private pile-up library (below); truth preset `applyTruthPreset(fragment="TenTau_E15to500_Eta3p1_pythia8_cfi", keepStableSpectators=True)` at DIGI and RECO |
| e/gamma guns | `eg` | `Pythia8PtGun`: e+e- and gamma gamma pairs, 1 < pT < 60 GeV, 1.6 < abs(eta) < 2.9, from the vertex | none (private gun) | truth preset `preset="gun", seedPdgIds=[11, -11, 22], keepStableSpectators=True` at DIGI (the c2 DIGI, which the dumps and probes read); the supercluster legs read the c1 DIGI of chains 0-99, which has the full graph |
| QCD | `qcd` | `QCD_Pt15To7000_Flat_14TeV_TuneCP5_cff` | 37851.88 (PU200), 37651.88 (no PU) | private pile-up library; truth preset `preset="full", seedPdgIds=[2212], seedParentDepth=0, keepStableSpectators=True` at DIGI |
| QCD, high pT-hat | `qcdhi` | `QCD_Pt15To7000_Flat_14TeV_TuneCP5_cff` with `PhaseSpace:pTHatMin 150`, `pTHatMax 1000`, `bias2Selection on`, `bias2SelectionPow 6`, `bias2SelectionRef 150` | 37851.88 with these generator settings | as QCD |
| pile-up library (c1) | `mb` | `MinBias_14TeV_pythia8_TuneCP5_cfi` | GEN-SIM step of 37640.88 (the PU input of 378xx) | private: 440 files of 250 events; files 0-319 (80,000 events) mix the training chains, files 320-439 the validation chains (chain % 5 == 4) |
| photon guns (regression only) | - | `SingleGammaE50_cfi` changed to two photons per event, 1.6 < abs(eta) < 2.8, E 20-500 GeV (PU200, PU0) and 400-1200 GeV (PU0) | none (private gun) | PU200 with half of the RelVal library `RelValMinBias_14TeV` D127 of CMSSW_20_0_0 (48 of its 95 files, the odd lines of the list) |

### Chains per model

| model | sample chains |
|---|---|
| mixture PIDs | probes: TenTau 200-238 (32, chain % 5 != 4), e/gamma 100-115, QCD 40-55, high-pT QCD 0-38 (32, chain % 5 != 4) |
| hypothesis, species | dumps: TenTau 862 of chains 0-1127, e/gamma 236 (64-299) |
| neutral share | dumps: TenTau 862, QCD 252 (24-299 without 100-123), high-pT QCD 64 (0-78) |
| supercluster score | e/gamma 0-31, 64-99 |
| supercluster type score | e/gamma 0-31, 64-99, TenTau 1, 6, ..., 196 and 3, 8, ..., 198 |
| energy regression | photon guns: PU200 files 100, 101, 102, 104; PU0 3000 events; high energy 500 events |
| hadronic calibration | TenTau 200-248 (40), QCD 24-55 (32) |

The productions: c1 = `ticlv6prod` (TenTau, e/gamma 0-99, the pile-up library), c2 = `ticlv6prod_c2` (e/gamma 100-299,
QCD), c3 = `ticlv6prod_c3` (high-pT QCD). The DIGI of the e/gamma chains 0-99 is in c2 too, from the c1 GEN-SIM. The
code of c1, c2 and c3 is `ticlv6-modifier` e6008d0379b in CMSSW_20_1_X_2026-10-04-0000.

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

The shell drivers (`*.sh`) find the other scripts of this repository relative to their own path. They read the
locations of a run from environment variables:

| variable | meaning | run rt3p2 |
|---|---|---|
| `RUNDIR` | run folder: the generated blocks (`blocks/`), the claims and one folder `<TAG>/` per retrain with its queues, outputs and markers | `/cms-hlt-nfs/user/fpantale/ticlv6tune/rt3` |
| `AREA` | CMSSW area of the RECO jobs (with `cmssw/` and the patch), into which the new models are installed | the area `rt3` (below) |
| `DATA` | root of the productions c1, c2, c3 (DIGI files) | `/cms-hlt-nfs/user/fpantale` (the default) |
| `TRAIN_ENV` | CMSSW area whose python has torch, onnx and xgboost | `/shared/fpantale/retrain_cs/CMSSW_20_1_0_pre4` (the default) |

The queues of a run are in `common/queues` (probe 96, dumps 1414, supercluster legs 148). The python scripts read the
data paths from environment variables or arguments.

## Order of a full retrain

1. `common/setup_tag.sh`: blocks, queues and folders of one run.
2. Probe chains, then the mixture PIDs (`showerShapePID/`).
3. Dump chains with the new mixture PIDs, then the interpretation models (`interpretation/`).
4. Supercluster legs with all new models, then the score and type models (`egammaSuperClusters/score_type/`) and,
   with `SC_REF`, the score cuts at the fake count of the old model.
   `common/rest.sh` runs step 4, installs the models and runs the validation leg.

In a GPU pod of the NGT cluster, `common/pod_start.sh` runs steps 2 and 3 (pools and training) in cmssw-el8.

Each step waits for the end marker of the step before it, so all steps can start at once. See `common/README.md`.

Run rt3p2 did not retrain the energy regression and the hadronic calibration.
