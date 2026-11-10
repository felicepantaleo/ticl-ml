# C++ against python for the final-level shape PID: P(photon) stored by ticlTracksterInterpretations on the final
# tracksters (table of extract_fin.py) against the mapped ONNX model on the same 16 inputs, for raw energy >= 1 GeV.
# usage: parity.py <fin table> <mapped onnx>   (cmsenv python; torch is not loaded, onnxruntime is)
import sys
import numpy as np
import onnxruntime as ort
import uproot

a = uproot.open(sys.argv[1])["fin"].arrays(library="np")
E = a["rawE"]; eig = np.maximum(a["eig"], 1e-6); ax = a["axis"]; eta = a["eta"]
X = np.column_stack([np.log1p(E), a["rawEmE"] / np.maximum(E, 1e-6), np.abs(eta), a["absZ"], np.log1p(a["nLC"]),
                     np.log(eig), eig[:, 0] / eig.sum(1), a["sig"], a["sigPCA"],
                     np.abs(np.arcsinh(ax[:, 2] / np.maximum(np.hypot(ax[:, 0], ax[:, 1]), 1e-6)) - eta)]).astype(np.float32)
m = E >= 1
s = ort.InferenceSession(sys.argv[2], providers=["CPUExecutionProvider"]).run(None, {"input": X[m]})[0].reshape(-1)
stored = a["pid"][m, 0]
d = np.abs(np.clip(s, 0, 1) - stored)
print(f"final tracksters {len(E)}, E >= 1: {m.sum()}; max |python - C++| {d.max():.2e}, 99.9% quantile {np.quantile(d, 0.999):.2e}; "
      f"P(chHad) = 1 - P(photon): {np.abs(a['pid'][m, 4] - (1 - stored)).max():.2e}")
