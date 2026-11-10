# Interpretation models

| model | CMSSW parameter | input | output |
|---|---|---|---|
| `hypothesis_mlp_v3.onnx` | `ticlTracksterInterpretations.arbitrationModelFile` | "features" [N, 32] (`RecoTICL/Interpretation/interface/ModelFeatures.h`) | "logit" [N], calibrated weight of a hypothesis |
| `neutralSpecies_mlp_v3.onnx` | `ticlTracksterInterpretations.neutralModelFile` | "features" [N, 12] | "logits" [N, 3]: photon, pi0, neutral hadron |
| `neutralShare_mlp_v2.onnx` | `ticlCandidate.neutralShareModel` | "features" [N, 31] | "fraction" [N], the energy share that a neutral candidate keeps |

## Dumps

One pass of 1414 chains (queue `<TAG>/queue_<TAG>_dumps.txt`, lines `m6 <sample> <chain>`): TenTau 862, e/gamma 236,
QCD 252, high-pT QCD 64. Each chain is a RECO job (`common/job3.sh m6 ...`) with the block `common/blocks/m6s_p2.py`
(template `common/blocks/m6s.py`) and the new mixture PIDs:

- `dumpHypotheses = True` (`cmssw/patches/0001-study-only-hypothesis-dump.patch`) and `HypothesisTruthDumper`
  (`cmssw/CornettoStudy`). The dumper writes the tree `hyps` (one row per hypothesis with the 32 model features and
  the truth label), the tree `pairs` (conflicting hypotheses) and the tree `neutrals` (12 species inputs and the truth
  species of each neutral final trackster).
- `CandidateEnergyProbe` (`cmssw/CornettoStudy`) gives the rows of the share model. The job runs without the share
  correction (`cmssw/patches/neutralShare_ones_study.onnx` gives 1) and without the minimum energy of the neutral
  candidates. The adaptive trackster associator runs without the pT floor (`truthBranchTracksterAssociatorsNoFloor`).

    TAG=rt3p2 STEP=dumps AREA=<cmssw area> WORK=<local dir> bash common/start_pool.sh 0 16 16

`start_train.sh` waits for all dump chains, writes the list of chain folders (`CHAINDIRS`) and runs `train_m6.sh`
with `PART=hyp` and `PART=share` in parallel.

    TAG=rt3p2 bash interpretation/start_train.sh

## Splits

By chain id: fit id % 5 in {0, 1, 2}, early stopping and calibration (es) id % 5 == 3. The rows with id % 5 == 4 are
not used. `combine_tables.py` adds the offset 10000 k to the chain ids of input k, so TenTau and e/gamma chains stay
separate.

## Hypothesis model (`PART=hyp`, TenTau + e/gamma)

1. `hypothesis/prepare_c2.py <sample> <out>`: tables `fit.npz`, `es.npz` from `hyps` and `neutrals`. The label rules
   for muons, neutral hadrons and charged hadrons are in the header.
2. `hypothesis/prepare_pairs.py`: same-track pairs (muon, electron, charged hadron, recovery) where exactly one
   hypothesis has the type of the track particle.
3. `hypothesis/combine_tables.py`: one table set from TenTau and e/gamma.
4. `hypothesis/prepare_cpairs.py`: conflict pairs (one right, one wrong) and `pairs_both` (conflict and same-track
   pairs).
5. `hypothesis/train_d2.py`, 4 seeds: MLP 64-64 SiLU, AdamW 2e-3, early stopping on es. Loss: pointwise loss plus a
   pairwise term softplus(logit_wrong - logit_right) with weight 0.3 (`PAIRW=0.3 PAIRSEL=all PAIRFILE=pairs_both`).
   Platt calibration on es (`PLATT=global`).
6. `hypothesis/ens_export.py <tables> <out.onnx> 0 2=0.5 <4 seeds>`: one ONNX with the mean of the 4 calibrated
   outputs, plus an offset of 0.5 for type 2 (photon). The producer runs it with `arbitrationPlattA` 1 and
   `arbitrationPlattB` 0.

## Neutral species model (`PART=hyp`)

`species/train_species_m6.py <tables> <out.onnx>`: the neutral rows of the combined tables (raw E > 1 GeV, dominant
species share >= 0.5). 12 inputs, MLP 64-64 SiLU, 3 logits, cross entropy, AdamW 2e-3, weight decay 1e-4,
ReduceLROnPlateau 0.5/3, batch 2048, patience 15, at most 300 epochs. Uses a GPU.

## Neutral share model (`PART=share`, TenTau + QCD + high-pT QCD)

1. `share/prepare_share_m6.py` (`M6ROOT` = dump root, `SHAREOUT` = output, `CHAINDIRS`): one `.npz` per chain with the
   31 features and the adaptive target (the footprint energy of the best truth particle over the footprint truth
   energy).
2. `share/train_share_m6.py <out.onnx>` (`SHARETABLES`): MLP 64-64 SiLU with a sigmoid output, unweighted MSE, AdamW
   2e-3, weight decay 1e-4, ReduceLROnPlateau 0.5/3, batch 2048, patience 10, at most 200 epochs. Uses the CPU.

The candidate producer keeps the full trackster energy for a neutral candidate with pT >= `neutralShareMinPt`
(default 10 GeV on `ticlv6-modifier`).

## Run rt3p2

`logs/`:
- driver logs (`hyp_driver.log`, `share_driver.log`);
- per seed: `arb_fc_both03g_s*.json` with the Platt a, b, the es loss and the row counts;
- `ensemble.log`, `species.log`, `share_model.log`.

Numbers:
- Hypothesis rows: fit 17.8 M, es 6.2 M.
- Species: 26.0 M rows, es loss 0.3636.
- Share: 836 fit chains, 26.9 M rows, 290 es chains, es MSE 0.01995.

A run of `ens_export.py` of this repository on the 4 stored seeds gave the deployed hypothesis file again (md5
7558c581, 2026-11-10).
