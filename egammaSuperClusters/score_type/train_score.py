#!/usr/bin/env python3
"""Train the TICL e/gamma supercluster score (xgboost) on the truth probe rows of the basep2 legs and export it to ONNX.

Rows: superclusters of ticlEGammaSuperClusters (kind 0, coll 0) with the 35 score features
(scoreFeatures of RecoTICL/ParticleFlow/interface/EGammaSuperClusterFeatures.h): x8..x38 and x40..x43 of the probe;
for a probe with 40 columns (before the track features), features 31-34 are made from the same quantities: the type
of the e/gamma trackster row (x0 of kind 4), x39 and x5 over the raw transverse energy.
Label 1: the signal (gun) deposit is at least 0.5 of the truth deposit. Label 0: the signal deposit is below 0.5 and no
gun particle is within dR 0.2 (a fake of claude-reconstruction tools/egamma-gun-ab.py). Other rows are not used.
Chains: training --train-chains (default 32-89), early stopping on --val-chains (default 90-99); comma lists of
ranges (for example 0-31,64-89).
usage: train_score.py <legdir>/eg <leg> <out dir> [--depth 6] [--rounds 2000] [--eta 0.05] [--threads 16]
Writes <out>/score.json (xgboost), <out>/score.onnx (input "input" [n, 31] float, output "score" [n] float = P(label 1)),
<out>/train.log.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import uproot
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ana_diag import gunDR  # noqa: E402

NF = 35


def rows(legdir, leg, chains):
    X, y, meta = [], [], []
    for c in chains:
        d = f"{legdir}/{leg}/{c:06d}"
        if not os.path.exists(f"{d}/done"):
            continue
        f = uproot.open(f"{d}/ntuple.root")
        o = f["egProbe/obj"].arrays(library="np")
        g = f["egProbe/gun"].arrays(library="np")
        t = {k: v[o["kind"] == 4] for k, v in o.items()}
        m = (o["kind"] == 0) & (o["coll"] == 0)
        o = {k: v[m] for k, v in o.items()}
        o["feat"] = features(o, t)
        dr = gunDR(o, g)
        sig = o["sigDep"] >= 0.5 * np.maximum(o["dep"], 1e-9)
        bad = ~sig & (dr > 0.2)
        use = sig | bad
        X.append(o["feat"][use])
        y.append(sig[use].astype(np.int32))
        meta.append(np.stack([np.full(use.sum(), c), o["rawE"][use] / np.cosh(o["eta"][use]), np.abs(o["eta"][use])], 1))
    return np.concatenate(X).astype(np.float32), np.concatenate(y), np.concatenate(meta)


def features(o, t):
    """The NF score features of the supercluster rows o; t: the e/gamma trackster rows of the same file."""
    if o["x"].shape[1] >= 44:
        return np.concatenate([o["x"][:, 8:39], o["x"][:, 40:44]], 1)
    ek = lambda a: a["run"].astype(np.int64) * 10**12 + a["lumi"].astype(np.int64) * 10**7 + a["event"].astype(np.int64)
    tk = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(ek(t), t["idx"]))}
    j = np.array([tk.get((int(a), int(b)), -1) for a, b in zip(ek(o), o["x"][:, 7])], dtype=int)
    typ = np.where(j >= 0, t["x"][np.maximum(j, 0), 0] if len(t["kind"]) else 0, 0)
    rawEt = np.maximum(o["rawE"] / np.cosh(o["eta"]), 1e-3)
    extra = np.stack([typ == 11, typ == 111, o["x"][:, 39], o["x"][:, 5] / rawEt], 1).astype(np.float32)
    return np.concatenate([o["x"][:, 8:39], extra], 1)


def to_onnx(bst, out):
    import onnx
    import onnxmltools
    from onnx import TensorProto, helper
    from onnxmltools.convert.common.data_types import FloatTensorType
    m = onnxmltools.convert_xgboost(bst, initial_types=[("input", FloatTensorType([None, NF]))], target_opset=15)
    g = m.graph
    prob = [o.name for o in g.output if "prob" in o.name.lower()]
    if not prob:
        raise SystemExit(f"no probability output in {[o.name for o in g.output]}")
    if not any(o.domain in ("", "ai.onnx") for o in m.opset_import):
        m.opset_import.append(helper.make_opsetid("", 15))
    g.initializer.append(helper.make_tensor("col1", TensorProto.INT64, [], [1]))
    g.node.append(helper.make_node("Gather", [prob[0], "col1"], ["score"], axis=1))
    del g.output[:]
    g.output.append(helper.make_tensor_value_info("score", TensorProto.FLOAT, [None]))
    onnx.checker.check_model(m)
    onnx.save(m, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("legdir")
    ap.add_argument("leg")
    ap.add_argument("out")
    ap.add_argument("--train-chains", default="32-89")
    ap.add_argument("--val-chains", default="90-99")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--drop", default="", help="comma list of feature indices that the model does not read")
    a = ap.parse_args()
    rng = lambda s: [c for p in s.split(",") for c in range(int(p.split("-")[0]), int(p.split("-")[-1]) + 1)]
    os.makedirs(a.out, exist_ok=True)
    Xt, yt, mt = rows(a.legdir, a.leg, rng(a.train_chains))
    Xv, yv, mv = rows(a.legdir, a.leg, rng(a.val_chains))
    drop = [int(x) for x in a.drop.split(",") if x]
    Xt[:, drop] = 0.
    Xv[:, drop] = 0.
    log = open(f"{a.out}/train.log", "w")
    msg = (f"train rows {len(yt)} (signal {yt.sum()}), chains {len(np.unique(mt[:, 0]))}; "
           f"val rows {len(yv)} (signal {yv.sum()}), chains {len(np.unique(mv[:, 0]))}")
    print(msg); log.write(msg + "\n")
    dt = xgb.DMatrix(Xt, label=yt)
    dv = xgb.DMatrix(Xv, label=yv)
    params = dict(objective="binary:logistic", eval_metric="logloss", max_depth=a.depth, eta=a.eta, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=5, nthread=a.threads, tree_method="hist", seed=1)
    bst = xgb.train(params, dt, a.rounds, evals=[(dt, "train"), (dv, "val")], early_stopping_rounds=100,
                    verbose_eval=200)
    bst = bst[: bst.best_iteration + 1]
    bst.save_model(f"{a.out}/score.json")
    pv = bst.predict(dv)
    msg = f"best iteration {bst.num_boosted_rounds()}, val logloss {-np.mean(yv * np.log(pv + 1e-9) + (1 - yv) * np.log(1 - pv + 1e-9)):.4f}"
    print(msg); log.write(msg + "\n")
    gain = bst.get_score(importance_type="gain")
    top = sorted(gain.items(), key=lambda kv: -kv[1])[:15]
    msg = "gain " + ", ".join(f"{k}:{v:.0f}" for k, v in top)
    print(msg); log.write(msg + "\n")
    to_onnx(bst, f"{a.out}/score.onnx")
    json.dump(dict(train=a.train_chains, val=a.val_chains, depth=a.depth, eta=a.eta, rounds=bst.num_boosted_rounds(),
                   nf=NF, drop=drop), open(f"{a.out}/meta.json", "w"))
    print(f"written {a.out}/score.onnx")
    return 0


if __name__ == "__main__":
    sys.exit(main())
