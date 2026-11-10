#!/usr/bin/env python3
"""Train the signal score of the TICL e/gamma superclusters for the type of the photon candidates (xgboost, the 35
score features), with a truth label: 1 when the signal deposit is at least 0.5 of the truth deposit of the
supercluster, 0 else (pile-up dominated). Rows: all superclusters of ticlEGammaSuperClusters of legs without the score
cut, from the e/gamma guns (signal = the gun particles) and from TenTau (signal = the tau decay products, for example
the photons of the neutral pions). Training: eg 0-31 and 64-89 and TenTau chains 1, 6, ..., 196; early stopping: eg 90-99 and
TenTau 3, 8, ..., 198. The validation chains (eg 32-63, TenTau 2, 7, ..., 197) are never read.
usage: train_type.py <eg legdir> <eg leg> <tt legdir> <tt leg> <out dir> [--threads 16] [--tt-weight 1]
Writes <out>/score.json and <out>/score.onnx (input "input" [n, 35], output "score" [n]).
"""
import argparse
import os
import sys

import numpy as np
import uproot
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_score import features, to_onnx  # noqa: E402


def rows(legdir, leg, chains):
    X, y = [], []
    for c in chains:
        d = f"{legdir}/{leg}/{c:06d}"
        if not os.path.exists(f"{d}/done"):
            continue
        o = uproot.open(f"{d}/ntuple.root")["egProbe/obj"].arrays(library="np")
        t = {k: v[o["kind"] == 4] for k, v in o.items()}
        m = (o["kind"] == 0) & (o["coll"] == 0)
        s = {k: v[m] for k, v in o.items()}
        X.append(features(s, t))
        y.append((s["sigDep"] >= 0.5 * np.maximum(s["dep"], 1e-9)).astype(np.int32))
    return np.concatenate(X).astype(np.float32), np.concatenate(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("egdir")
    ap.add_argument("egleg")
    ap.add_argument("ttdir")
    ap.add_argument("ttleg")
    ap.add_argument("out")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--tt-weight", type=float, default=1.)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    Xe, ye = rows(a.egdir, a.egleg, [*range(0, 32), *range(64, 90)])
    Xt, yt = rows(a.ttdir, a.ttleg, range(1, 197, 5))
    Xev, yev = rows(a.egdir, a.egleg, range(90, 100))
    Xtv, ytv = rows(a.ttdir, a.ttleg, range(3, 199, 5))
    print(f"train eg {len(ye)} (signal {ye.sum()}), tt {len(yt)} (signal {yt.sum()}); val eg {len(yev)}, tt {len(ytv)}")
    X = np.concatenate([Xe, Xt]); y = np.concatenate([ye, yt])
    w = np.concatenate([np.ones(len(ye)), np.full(len(yt), a.tt_weight)])
    Xv = np.concatenate([Xev, Xtv]); yv = np.concatenate([yev, ytv])
    p = dict(objective="binary:logistic", eval_metric="logloss", max_depth=6, eta=0.05, subsample=0.8,
             colsample_bytree=0.8, min_child_weight=5, nthread=a.threads, tree_method="hist", seed=1)
    b = xgb.train(p, xgb.DMatrix(X, label=y, weight=w), 2000, evals=[(xgb.DMatrix(Xv, label=yv), "val")],
                  early_stopping_rounds=100, verbose_eval=200)
    b = b[: b.best_iteration + 1]
    b.save_model(f"{a.out}/score.json")
    to_onnx(b, f"{a.out}/score.onnx")
    for name, Xs, ys in (("eg val", Xev, yev), ("tt val", Xtv, ytv)):
        pv = b.predict(xgb.DMatrix(Xs))
        for eff in (0.98, 0.95, 0.9):
            thr = np.quantile(pv[ys == 1], 1 - eff)
            print(f"  {name}: signal eff {eff}: cut {thr:.3f}, pile-up superclusters kept {(pv[ys == 0] >= thr).mean():.4f}")
    gain = sorted(b.get_score(importance_type="gain").items(), key=lambda kv: -kv[1])[:12]
    print("gain " + ", ".join(f"{k}:{v:.0f}" for k, v in gain))
    return 0


if __name__ == "__main__":
    sys.exit(main())
