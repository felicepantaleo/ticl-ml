#!/usr/bin/env python3
"""Score cuts of a new supercluster score model at the fake count of a reference model and its cuts.

For each range of raw supercluster Et (rawE / cosh(eta)), from the minimum to the split and at or above the split, the
cut of the new model is the smallest cut on a grid of 0.005 that keeps no more fake superclusters than the reference
model at its cut, on the validation chains of the supercluster legs (default 90-99, which the score training uses only for early stopping).

usage: equal_fake_cuts.py <legdir>/eg <leg> <reference onnx> <reference cut below split> <reference cut above split>
                          <new onnx> [--chains 90-99] [--min-et 2] [--split 10]
The last line is "CUTS <cut below split> <cut above split>".
"""
import argparse
import os
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_score import rows  # noqa: E402


def chain_range(text):
    lo, hi = (int(v) for v in text.split("-"))
    return range(lo, hi + 1)


def score(path, X):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return s.run(None, {s.get_inputs()[0].name: X})[0].ravel()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("legdir")
    p.add_argument("leg")
    p.add_argument("ref")
    p.add_argument("ref_low", type=float)
    p.add_argument("ref_high", type=float)
    p.add_argument("new")
    p.add_argument("--chains", type=chain_range, default=chain_range("90-99"))
    p.add_argument("--min-et", type=float, default=2.0)
    p.add_argument("--split", type=float, default=10.0)
    a = p.parse_args()
    X, y, meta = rows(a.legdir, a.leg, a.chains)
    et = meta[:, 1]
    ref, new = score(a.ref, X), score(a.new, X)
    grid = np.round(np.arange(0.0, 1.0001, 0.005), 3)
    cuts = []
    for name, sel, rc in (("low", (et >= a.min_et) & (et < a.split), a.ref_low), ("high", et >= a.split, a.ref_high)):
        sig, fake = sel & (y == 1), sel & (y == 0)
        nref = int(np.sum(ref[fake] >= rc))
        ok = [c for c in grid if np.sum(new[fake] >= c) <= nref]
        c = float(min(ok)) if ok else 1.0
        cuts.append(c)
        print(f"Et {name} ({int(sig.sum())} signal, {int(fake.sum())} fake): reference cut {rc:.4f} eff {np.mean(ref[sig] >= rc):.4f} "
              f"fakes {nref}; new cut {c:.3f} eff {np.mean(new[sig] >= c):.4f} fakes {int(np.sum(new[fake] >= c))}")
    print(f"CUTS {cuts[0]:.3f} {cuts[1]:.3f}")


if __name__ == "__main__":
    main()
