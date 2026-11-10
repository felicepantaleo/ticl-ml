# PyTorch models of the heterogeneous inference

With `--procModifiers ticl_v6,ticl_v6_heterogeneous`, two models run on the device with PyTorchAlpaka. They read
TorchScript files, not ONNX. The files are on the branch `hgcalOnly-workflow` of felicepantaleo/cmssw (head
5a3ff681d99, 2026-11-09), the workflow 0.432 of the HGCAL DPG hackathon.

| file | CMSSW path | producer | md5 | ONNX model it replaces |
|---|---|---|---|---|
| `mixPID_hi_v2.pt` | `RecoTICL/Inference/data/showerShapePID/mixPID_hi_v2.pt` | `TracksterShowerShapeInferenceProducer@alpaka` (device PID of the CLUEstering tracksters, `TICLConfig.device_pid()`) | 14aac6c2 | `mixPID_hi_v2.onnx` (`../showerShapePID`) |
| `supercls_v3.pt` | `RecoTICL/Superclustering/data/supercls_v3.pt` | `SuperclusteringDNNProducer@alpaka` (`superclusteringDNNHeterogeneous_cff.py`) | 04ee12c5 | `RecoHGCal/TICL/data/superclustering/supercls_v3.onnx` (cms-data) |

Both models are exports of an ONNX model, with no new training. The training of each model is the training of its
ONNX model.

- `mixPID_hi_v2.pt`: `exportShowerShapePIDToTorchScript.py` (from `RecoTICL/Inference/test`) writes the ONNX graph op
  for op as a TorchScript module. The weights and constants come from the ONNX file. The module ends with the clamp to
  [0, 1] that `TracksterInferenceByShowerShape` applies on the host. Input [N, 22], output "pid" [N, 8] in
  `ticl::Trackster::ParticleType` order. The script saves the module only if it reproduces onnxruntime within
  `--tolerance` (1e-5) on 20000 inputs. Export of 2026-11-09: 2.29e-6.

      python3 exportShowerShapePIDToTorchScript.py RecoTICL/Inference/data/showerShapePID/mixPID_hi_v2.onnx

  The mixture PID of the final tracksters (`mixPID_fin_v2`) stays on the host with ONNX.
- `supercls_v3.pt`: the superclustering DNN of TICLv5 (`supercls_v3.onnx`, input [N, 18], output [N, 1]), exported by
  the authors of felicepantaleo/cmssw#75 (Wahid Redjeb, Max Marriott-Clarke). The file is byte-identical to the file of
  that PR (head c7537334d0f). The PR has no export script for this model, and the model was not trained again for
  TICLv6.

`check_torchscript.py <model.pt> <model.onnx> ...` compares a TorchScript file with its ONNX model on 4096 random
inputs. Result of 2026-11-10 (CMSSW_20_1_X_2026-11-08-2300, CPU):

| file | max abs(torch - onnxruntime) |
|---|---|
| `mixPID_hi_v2.pt` | 1.25e-6 |
| `supercls_v3.pt` | 4.47e-6 |

A new export of `mixPID_hi_v2.onnx` gives the same outputs, but not the same bytes: the TorchScript archive is not
byte-reproducible. Compare outputs, not md5.

On the hackathon branch, the device PID against the host PID (serial backend, 50 PU200 events, 24530 tracksters
>= 1 GeV) differs by at most 1.85e-5 per class. The 37819 linked tracksters and the 243 superclusters are identical.
