#!/usr/bin/env python3
"""CMSSW against offline inference: for the superclusters of a dump2npz.py file (dumper on the main supercluster of a
B job), compare the CMSSW energy with raw energy times the offline model output on the dumped features.
Usage: check_parity.py <file.npz> <label> <model>"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare3_predict import predict  # noqa: E402

f, label, model = sys.argv[1:4]
d = np.load(f)
x = d[f"{label}_x"].astype(np.float32)
ok = np.all(np.isfinite(x), axis=1)
e = d[f"{label}_e"][ok]
r = d[f"{label}_raw"][ok] * predict(model, x[ok])
dev = np.abs(r / e - 1)
print(f"{ok.sum()} superclusters: max |offline/CMSSW - 1| = {dev.max():.2e}, 99.9% quantile {np.quantile(dev, 0.999):.2e}")
