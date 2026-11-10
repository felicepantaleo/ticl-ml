#!/usr/bin/env python3
"""Compare TorchScript models with the ONNX models they replace: max |torch - onnxruntime| on random inputs.
For a mixture PID the inputs are around the standardisation of the model (mean + N(0, 1) x std).
usage: check_torchscript.py <model.pt> <model.onnx> [<model.pt> <model.onnx> ...]   (cmsenv python, CPU)
"""
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper


def to_torch(a):
    # the CMSSW PyTorch has no NumPy bridge
    return torch.tensor(a.tolist(), dtype=torch.float32)


def check(pt, ox, n=4096, seed=1):
    model = torch.jit.load(pt).eval()
    sess = ort.InferenceSession(ox, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    x = np.random.default_rng(seed).normal(0, 1, (n, inp.shape[1])).astype(np.float32)
    init = {i.name: numpy_helper.to_array(i) for i in onnx.load(ox).graph.initializer}
    if "em.net.mean" in init:
        x = (init["em.net.mean"] + x * init["em.net.std"]).astype(np.float32)
    ref = sess.run(None, {inp.name: x})[0]
    with torch.no_grad():
        out = model(to_torch(x))
    if isinstance(out, (tuple, list)):
        out = out[0]
    out = np.array(out.tolist(), dtype=np.float32).reshape(ref.shape)
    print(f"{pt}: input {inp.name} {inp.shape}, output {ref.shape}, max |torch - onnx| {np.max(np.abs(out - ref)):.3g}")


args = sys.argv[1:]
if not args or len(args) % 2:
    sys.exit(__doc__)
for k in range(0, len(args), 2):
    check(args[k], args[k + 1])
