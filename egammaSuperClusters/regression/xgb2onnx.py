#!/usr/bin/env python3
"""Convert an xgboost regression of log(E_gen / E_raw) (train3.py --model xgb) to ONNX with the interface of the
CMSSW producer: input "input" [n, F] float, output "ratio" [n] = exp of the model output.
Usage: xgb2onnx.py <model.json> <out.onnx>"""
import sys

import numpy as np
import onnx
import onnxmltools
import xgboost as xgb
from onnx import TensorProto, helper
from onnxmltools.convert.common.data_types import FloatTensorType

fin, fout = sys.argv[1:3]
bst = xgb.Booster()
bst.load_model(fin)
nf = bst.num_features()
import json
import os
cols = None
if os.path.exists(fin + ".cols"):
    meta = json.load(open(fin + ".cols"))
    cols, nf_in = meta["cols"], meta["nf"]
cfg = __import__("json").loads(bst.save_config())
base = float(cfg["learner"]["learner_model_param"]["base_score"])
m = onnxmltools.convert_xgboost(bst, initial_types=[("input", FloatTensorType([None, nf]))], target_opset=15)
if not any(o.domain in ("", "ai.onnx") for o in m.opset_import):
    m.opset_import.append(helper.make_opsetid("", 15))
g = m.graph
out = g.output[0].name
g.node.append(helper.make_node("Exp", [out], ["ratio2d"]))
g.node.append(helper.make_node("Flatten", ["ratio2d"], ["ratioflat"], axis=0))
g.node.append(helper.make_node("Squeeze", ["ratioflat", "axes0"], ["ratio"]))
g.initializer.append(helper.make_tensor("axes0", TensorProto.INT64, [1], [0]))
del g.output[:]
g.output.append(helper.make_tensor_value_info("ratio", TensorProto.FLOAT, [None]))
if cols is not None:
    # The model reads the columns cols of an input with nf_in columns.
    inp = g.input[0]
    old = inp.name
    g.input.remove(inp)
    g.input.insert(0, helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, nf_in]))
    for node in g.node:
        for k, name in enumerate(node.input):
            if name == old:
                node.input[k] = "input_cols"
    g.initializer.append(helper.make_tensor("cols_idx", TensorProto.INT64, [len(cols)], cols))
    g.node.insert(0, helper.make_node("Gather", ["input", "cols_idx"], ["input_cols"], axis=1))
    nf = nf_in
onnx.checker.check_model(m)
onnx.save(m, fout)
x = np.random.default_rng(1).normal(size=(5, nf)).astype(np.float32)
print(f"written {fout}: {nf} features, base_score {base}")
