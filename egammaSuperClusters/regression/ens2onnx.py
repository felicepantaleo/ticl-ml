#!/usr/bin/env python3
"""Merge regression models (ONNX files with input "input" [n, F] and output "ratio" [n], or xgboost .json files of
train3.py) into one ONNX model with the same interface whose output is the weighted geometric mean of the member
outputs. Each member reads the first F_i features; the merged input has max F_i features.
Usage: ens2onnx.py <out.onnx> <model>[:weight] ..."""
import os
import subprocess
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, compose, helper

out, members = sys.argv[1], sys.argv[2:]
here = os.path.dirname(os.path.abspath(__file__))
models, weights = [], []
for spec in members:
    m, _, w = spec.partition(":")
    weights.append(float(w) if w else 1.)
    if m.endswith(".json"):
        tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False).name
        subprocess.run([sys.executable, os.path.join(here, "xgb2onnx.py"), m, tmp], check=True,
                       stdout=subprocess.DEVNULL)
        m = tmp
    models.append(onnx.load(m))
nfs = [mod.graph.input[0].type.tensor_type.shape.dim[1].dim_value for mod in models]
nf = max(nfs)
nodes, inits, opsets = [], [], {}
logs = []
nodes.append(helper.make_node("Constant", [], ["ens_starts"], value=helper.make_tensor("s", TensorProto.INT64, [1], [0])))
nodes.append(helper.make_node("Constant", [], ["ens_axes"], value=helper.make_tensor("a", TensorProto.INT64, [1], [1])))
for i, (mod, f) in enumerate(zip(models, nfs)):
    for o in mod.opset_import:
        opsets[o.domain] = max(opsets.get(o.domain, 0), o.version if o.domain not in ("", "ai.onnx") else 17)
    p = compose.add_prefix(mod, prefix=f"m{i}_")
    g = p.graph
    nodes.append(helper.make_node("Constant", [], [f"m{i}_ends"],
                                  value=helper.make_tensor("e", TensorProto.INT64, [1], [f])))
    nodes.append(helper.make_node("Slice", ["input", "ens_starts", f"m{i}_ends", "ens_axes"], [g.input[0].name]))
    nodes.extend(g.node)
    inits.extend(g.initializer)
    nodes.append(helper.make_node("Log", [g.output[0].name], [f"m{i}_logu"]))
    nodes.append(helper.make_node("Constant", [], [f"m{i}_w"],
                                  value=helper.make_tensor("w", TensorProto.FLOAT, [], [weights[i] / sum(weights)])))
    nodes.append(helper.make_node("Mul", [f"m{i}_logu", f"m{i}_w"], [f"m{i}_log"]))
    logs.append(f"m{i}_log")
nodes.append(helper.make_node("Sum", logs, ["ens_mean"]))
nodes.append(helper.make_node("Exp", ["ens_mean"], ["ratio"]))
graph = helper.make_graph(nodes, "ensemble", [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["n", nf])],
                          [helper.make_tensor_value_info("ratio", TensorProto.FLOAT, ["n"])], inits)
if "" not in opsets:
    opsets[""] = 17
model = helper.make_model(graph, opset_imports=[helper.make_opsetid(d, v) for d, v in opsets.items()])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, out)
print(f"written {out}: {len(models)} members, weights {weights}, {nf} features, opsets {opsets}")
