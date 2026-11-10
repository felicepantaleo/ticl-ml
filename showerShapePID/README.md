# Mixture PID (shower-shape identification)

Models `mixPID_hi_v2.onnx` (CLUEstering tracksters, `ticlTrackstersCLUEstering`) and `mixPID_fin_v2.onnx` (final
tracksters, `ticlTracksterInterpretations`). The CMSSW plugin is `TracksterInferenceByShowerShape` in mixture mode
(`RecoTICL/Inference`).

## What the model does

- Input "input" [N, 22]: the 16 trackster quantities of the shape PID (`pidstudy/shape/train_shape.py`, `features`)
  and 6 raw layer-cluster profile values (largest share, core shares at 2 and 5 cm, transverse rms, min abs(z),
  abs(z) extent).
- Network: MLP 128-128-64 SiLU with a softmax over 7 species (photon, electron, pi0, charged hadron, neutral hadron,
  muon, other).
- Target: the soft labels `frac[7]` of `StageTruthProbe`, which are the truth deposit fractions of the trackster by
  species.
- Loss: soft cross entropy. Each raw-energy bin gets the same total weight. AdamW 2e-3, weight decay 1e-4,
  ReduceLROnPlateau 0.5/3, at most 300 epochs.
- Rows: raw energy >= 1 GeV, with a truth deposit. Split: chain % 4 == 3 is validation.
- Map (`export_mix.py`): the EM sum s = photon + electron + pi0 is mapped per raw energy onto the scale of the CNN
  that the model replaces. The map matches the hadron quantiles (hadron: truth EM fraction < 0.5) in raw-energy bins,
  linear in log(1 + E) between the bin centres. The EM slots are scaled by g/s, the other slots by (1 - g)/(1 - s).
  So a threshold on photon + electron keeps the CNN hadron mis-identification in every energy bin. The output is the
  8 PID slots of a trackster.

## Steps

1. Probe chains (96 chains: TenTau 32, e/gamma 16, QCD 16, high-pT QCD 32; queue `<TAG>/queue_<TAG>_probe.txt`,
   lines `probe <sample> <chain>`). Each chain is a RECO job (`common/job3.sh probe ...`) with the block
   `common/blocks/probe_p2.py` (template `common/blocks/probe.py`):
   - `StageTruthProbe` (`cmssw/CornettoStudy`) writes the truth of every trackster at every stage (`frac`, `lcProf`).
   - An EDM output `tracksters.root` keeps the CLUEstering tracksters and the final tracksters.
   - The CNN PIDs run at both levels, because they give the scale of the map.

       TAG=rt3p2 STEP=probe AREA=<cmssw area> WORK=<local dir> bash common/start_pool.sh 0 16 16

2. Training (`train_mix.sh`, waits for the 96 probe chains, deadline 6 h):
   - `pidstudy/stage2/extract.py <chain dir>`: `tracksters.root` -> `ts_flat.root`, one row per trackster (trees `hi`
     and `fin`).
   - `pidstudy/mix/train_mix.py <hi|fin> <stem>` (`PROBE_DIR` = the probe folder): `<stem>.pt`, `<stem>.onnx`
     (7 fractions) and `<stem>_eval.json`.
   - `pidstudy/mix/export_mix.py <hi|fin> <stem>`: `<stem>_mapped.onnx` (the deployed file) and `<stem>_map.json`.

       TAG=rt3p2 bash showerShapePID/train_mix.sh

3. The dumps of `interpretation/` install `<stem>_mapped.onnx` as `mixPID_{hi,fin}_<TAG>.onnx` in the CMSSW area,
   because the interpretation models read the 8 PID slots.

## Reference files

`pidstudy/shape/shapePID_v1*`, `pidstudy/stage2/model/shapePIDFinal_v1*` and `pidstudy/emap/*` are the earlier shape
models (v1: one map over all energies; v2: an energy map). `train_mix.py` uses them only for the comparison in
`_eval.json`. `pidstudy/shape/train_shape.py` and `pidstudy/shape/emap.py` made them. `pidstudy/stage2/train_final.py`
made the v1 model of the final tracksters. `pidstudy/stage2/probe_block.py` and `extract_fin.py` belong to that study.
`pidstudy/mix/fold_pi0.py` is a test that was not adopted.

## Run rt3p2

`logs/`: training and export logs, `_eval.json` and `_map.json` of both levels. Parity ONNX against torch: hi
9.5e-7, fin 5.4e-7. A run of `export_mix.py hi` of this repository on the stored `.pt` and probe rows gave the
deployed file again (md5 49ea7f9a, 2026-11-10).
