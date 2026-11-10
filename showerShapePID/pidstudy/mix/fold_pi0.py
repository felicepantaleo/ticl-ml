#!/usr/bin/env python3
"""Folded copy of a deployed mixture-PID ONNX: the neutral_pion slot (3) is added to the photon slot (0) and set to 0,
so consumers that read photon + electron as the EM probability see the mapped EM sum. Output name stays "pid".
usage: fold_pi0.py <in.onnx> <out.onnx>"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper

m = onnx.load(sys.argv[1])
g = m.graph
assert [o.name for o in g.output] == ["pid"]
F = np.eye(8, dtype=np.float32)
F[3, 3] = 0.0
F[3, 0] = 1.0  # out[:, 0] = in[:, 0] + in[:, 3]
for n in g.node:
    n.output[:] = ["pid_unfolded" if o == "pid" else o for o in n.output]
g.initializer.append(numpy_helper.from_array(F, "pi0_fold"))
g.node.append(helper.make_node("MatMul", ["pid_unfolded", "pi0_fold"], ["pid"], name="fold_pi0_into_photon"))
onnx.checker.check_model(m)
onnx.save(m, sys.argv[2])
print("wrote", sys.argv[2])
