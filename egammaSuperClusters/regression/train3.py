#!/usr/bin/env python3
"""Train the energy regression of the TICL e/gamma superclusters on dump2npz.py files: the model predicts
log(E_gen / E_raw) with an L1 loss (the median), and the ONNX output "ratio" [n] is exp of it; input "input" [n, F]
holds the first F features of EGammaSuperClusterFeatures.h (or the columns given by --cols).
Usage: train3.py <out.onnx> <label> --data file.npz:weight:lumis ... [--nf F] [--model mlp|xgb] [--hidden 64,64]
       [--epochs N] [--seed S]
lumis is a "+"-separated list of luminosity blocks ("all": every block)."""
import argparse
import json

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("label")
p.add_argument("--data", nargs="+", required=True)
p.add_argument("--nf", type=int, default=19)
# With --cols, the model reads only these columns of the first nf features (the ONNX input keeps nf columns).
p.add_argument("--cols", default="")
p.add_argument("--model", default="mlp")
p.add_argument("--hidden", default="64,64")
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--max-ratio", type=float, default=3.0)
p.add_argument("--trees", type=int, default=2000)
p.add_argument("--huber", type=float, default=0.03)
p.add_argument("--depth", type=int, default=6)
p.add_argument("--threads", type=int, default=8)
args = p.parse_args()
np.random.seed(args.seed)

xs, ts, ws, evs = [], [], [], []
for spec in args.data:
    f, w, lumis = spec.split(":")
    d = np.load(f)
    x = d[f"{args.label}_x"][:, :args.nf]
    sel = np.all(np.isfinite(x), axis=1)
    if lumis != "all":
        sel &= np.isin(d["lumi"].astype(int), [int(v) for v in lumis.split("+")])
    with np.errstate(divide="ignore", invalid="ignore"):
        target = np.log(d["egen"] / d[f"{args.label}_raw"])
    # Superclusters far from the photon energy are other objects (a split shower or a conversion leg): out of the fit.
    sel &= np.isfinite(target) & (np.abs(target) < np.log(args.max_ratio))
    xs.append(x[sel])
    ts.append(target[sel])
    ws.append(np.full(sel.sum(), float(w)))
    evs.append(d["event"][sel] + 100000 * d["lumi"][sel])
    print(f"{f}: {sel.sum()} superclusters, weight {w}, lumis {lumis}")
x = np.concatenate(xs).astype(np.float32)
cols = [int(c) for c in args.cols.split(",")] if args.cols else list(range(x.shape[1]))
nf_in = x.shape[1]
x = x[:, cols]
target = np.concatenate(ts).astype(np.float32)
weight = np.concatenate(ws).astype(np.float32)
event = np.concatenate(evs).astype(np.int64)
# Split by event: events with number % 5 == 4 validate (early stopping only).
val = (event % 5) == 4
print(f"samples {len(x)}: train {np.sum(~val)}, validation {np.sum(val)}, features {x.shape[1]}")

if args.model == "mlp":
    import torch
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    mean = x[~val].mean(axis=0)
    std = x[~val].std(axis=0)
    std[std == 0] = 1.
    hidden = [int(h) for h in args.hidden.split(",")]

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
            self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
            self.register_buffer("cols", torch.tensor(cols, dtype=torch.int64))
            layers, n = [], x.shape[1]
            for h in hidden:
                layers += [torch.nn.Linear(n, h), torch.nn.SiLU()]
                n = h
            layers.append(torch.nn.Linear(n, 1))
            self.body = torch.nn.Sequential(*layers)

        def logratio(self, inp):
            return self.body((inp - self.mean) / self.std)[:, 0]

        def forward_full(self, inp):
            return torch.exp(self.logratio(torch.index_select(inp, 1, self.cols)))

        def forward(self, inp):
            return torch.exp(self.logratio(inp))

    net = Net()
    xt, yt, wt = (torch.tensor(a[~val]) for a in (x, target, weight))
    xv, yv, wv = (torch.tensor(a[val]) for a in (x, target, weight))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best, best_state = 1e9, None
    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 256):
            b = perm[i:i + 256]
            loss = torch.sum(wt[b] * torch.abs(net.logratio(xt[b]) - yt[b])) / torch.sum(wt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            lv = (torch.sum(wv * torch.abs(net.logratio(xv) - yv)) / torch.sum(wv)).item()
        if lv < best:
            best, best_state = lv, {k: v.clone() for k, v in net.state_dict().items()}
        if epoch % 100 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch}: validation L1 {lv:.5f}")
    net.load_state_dict(best_state)
    net.eval()
    print(f"best validation L1 {best:.5f}")
    net.forward = net.forward_full
    torch.onnx.export(net, torch.zeros(4, nf_in), args.out, input_names=["input"], output_names=["ratio"],
                      dynamic_axes={"input": {0: "n"}, "ratio": {0: "n"}}, opset_version=17, dynamo=False)
else:
    import xgboost as xgb
    dt = xgb.DMatrix(x[~val], label=target[~val], weight=weight[~val])
    dv = xgb.DMatrix(x[val], label=target[val], weight=weight[val])
    # Pseudo-Huber loss with a small slope: close to the L1 loss (the median); reg:absoluteerror of xgboost 1.7.5
    # diverges on this data (validation L1 grows from the 8th tree).
    params = dict(objective="reg:pseudohubererror", huber_slope=args.huber, eta=0.05, max_depth=args.depth,
                  subsample=0.8, colsample_bytree=0.9, min_child_weight=10, nthread=args.threads, seed=args.seed,
                  eval_metric="mae", base_score=float(np.median(target[~val])))
    bst = xgb.train(params, dt, args.trees, evals=[(dv, "val")], early_stopping_rounds=100, verbose_eval=200)
    print(f"best iteration {bst.best_iteration}, validation L1 {bst.best_score}")
    bst = xgb.train(params, dt, bst.best_iteration + 1)
    bst.save_model(args.out.replace(".onnx", ".json"))
    if args.cols:
        open(args.out.replace(".onnx", ".json") + ".cols", "w").write(json.dumps(dict(cols=cols, nf=nf_in)))
    print(json.dumps(dict(trees=bst.best_iteration if hasattr(bst, "best_iteration") else args.trees)))
print(f"written {args.out}")
