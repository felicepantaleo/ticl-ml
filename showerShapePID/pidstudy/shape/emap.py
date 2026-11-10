#!/usr/bin/env python3
"""Energy-dependent map of the shower-shape EM score onto the scale of the CNN that it replaces.

The v1 map g(s) matches the hadron quantiles of the score s and of the CNN over all energies. Then a threshold keeps the
CNN hadron mis-ID on average, but not per energy: above 20 GeV the mis-ID is 2 to 10 times the CNN one. This map
g(s, E) matches the hadron quantiles in bins of the raw energy and interpolates linearly in log(1 + E) between the bin
centres, so a threshold keeps the CNN hadron mis-ID in every energy bin. The network is the v1 network, unchanged.

level 1: CLUE3DHigh tracksters (pidstudy/probe, CNN/patternrecognition/id_v0, net shape/shapePID_v1.onnx)
level 2: final tracksters (pidstudy/stage2/probe, CNN/linking/id_v0, net stage2/model/shapePIDFinal_v1.onnx)
The map uses the train chains (chain % 4 != 3); the tables use the validation chains.
Output: <out>.onnx (input "input" [N, 16], output "em" [N]) and <out>_map.json.

usage: emap.py <1|2> <out stem>   (cmsenv python)
"""
import json
import os
import sys

import subprocess
import tempfile

import numpy as np
import torch

# Root of the pidstudy scripts and reference models (default: this repository).
PIDSTUDY = os.environ.get("PIDSTUDY", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, os.path.join(PIDSTUDY, "shape"))
sys.path.insert(0, os.path.join(PIDSTUDY, "stage2"))
from train_shape import Net, auc, features, to_np  # noqa: E402

EDGES = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100, 1e9]
QS = np.linspace(0, 1, 201)
MIN_HAD = 1000


def load_net(path):
    # onnx and torch in one process abort on a duplicate protobuf descriptor: read the weights in a subprocess.
    with tempfile.TemporaryDirectory() as d:
        code = ("import sys, numpy as np, onnx, onnx.numpy_helper as nh; "
                "np.savez(sys.argv[2], **{i.name: nh.to_array(i) for i in onnx.load(sys.argv[1]).graph.initializer})")
        subprocess.run([sys.executable, "-I", "-c", code, path, f"{d}/w.npz"], check=True)
        w = dict(np.load(f"{d}/w.npz"))
    net = Net(w["mean"], w["std"])
    net.body.load_state_dict({k[5:]: torch.tensor(v) for k, v in w.items() if k.startswith("body.")})
    return net.eval()


class EMapped(torch.nn.Module):
    """g(u, E) = sum_k w_k(t) g_k(u), u = the network logit, t = log(1 + E) = input column 0; g_k is the ramp sum of
    the knots of bin k (in the logit: the sigmoid crowds the hadron quantiles near 0 and float32 loses the steep slopes),
    clamped to its range; w_k are the tent functions on the bin centres c_k (a partition of unity, constant outside)."""

    def __init__(self, net, centres, sk, ck):
        super().__init__()
        self.net = net
        K, J = len(sk), max(len(s) for s in sk)
        knots, dslope, c0, c1 = np.zeros((K, J - 1)), np.zeros((K, J - 1)), np.zeros(K), np.zeros(K)
        for k in range(K):
            s = np.asarray(sk[k], np.float64)
            c = np.asarray(ck[k], np.float64)
            slope = np.diff(c) / np.maximum(np.diff(s), 1e-12)
            n = len(s) - 1
            knots[k, :n] = s[:-1]
            knots[k, n:] = s[-1]
            dslope[k, :n] = np.diff(np.concatenate([[0.0], slope]))
            c0[k], c1[k] = c[0], c[-1]
        c = np.asarray(centres, np.float64)
        lo = np.concatenate([[c[0] - 1.0], c[:-1]])
        hi = np.concatenate([c[1:], [c[-1] + 1.0]])
        f = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
        self.register_buffer("knots", f(knots))
        self.register_buffer("dslope", f(dslope))
        self.register_buffer("c0", f(c0))
        self.register_buffer("c1", f(c1))
        self.register_buffer("cen", f(c))
        self.register_buffer("lo", f(lo))
        self.register_buffer("hi", f(hi))
        self.register_buffer("first", f(np.eye(K)[0]))
        self.register_buffer("last", f(np.eye(K)[-1]))

    def forward(self, x):
        s = self.net.logit(x)[:, None, None]
        g = self.c0 + (torch.relu(s - self.knots) * self.dslope).sum(dim=2)
        g = torch.minimum(torch.maximum(g, self.c0), self.c1)
        t = x[:, :1]
        up = torch.clamp((t - self.lo) / (self.cen - self.lo), 0.0, 1.0)
        down = torch.clamp((self.hi - t) / (self.hi - self.cen), 0.0, 1.0)
        up = torch.maximum(up, self.first)
        down = torch.maximum(down, self.last)
        w = torch.minimum(up, down)
        return torch.clamp((w * g).sum(dim=1), 0.0, 1.0)


def numpy_map(s, t, centres, sk, ck):
    g = np.stack([np.interp(s, sk[k], ck[k]) for k in range(len(sk))], axis=1)
    c = np.asarray(centres)
    w = np.zeros_like(g)
    tc = np.clip(t, c[0], c[-1])
    j = np.clip(np.searchsorted(c, tc) - 1, 0, len(c) - 2)
    a = (tc - c[j]) / (c[j + 1] - c[j])
    w[np.arange(len(t)), j] = 1 - a
    w[np.arange(len(t)), j + 1] += a
    return np.clip((w * g).sum(1), 0, 1)


def main():
    level, out = int(sys.argv[1]), sys.argv[2]
    if level == 1:
        from train_shape import load
        A, nC = load()
        cnn = A["pCNNA"][:, 0] + A["pCNNA"][:, 1]
        net_path = os.path.join(PIDSTUDY, "shape/shapePID_v1.onnx")
        v1_map = json.load(open(os.path.join(PIDSTUDY, "shape/shapePID_v1_map.json")))
        v1_knots = (v1_map["score"], v1_map["cnn"])
        thresholds = (0.1, 0.3, 0.5, 0.6)
    else:
        from train_final import load
        A, nC = load()
        cnn = A["pid"][:, 0] + A["pid"][:, 1]
        net_path = os.path.join(PIDSTUDY, "stage2/model/shapePIDFinal_v1.onnx")
        v1_map = json.load(open(os.path.join(PIDSTUDY, "stage2/model/shapePIDFinal_v1_map.json")))
        v1_knots = (v1_map["score"], v1_map["linkCNN"])
        thresholds = (0.3, 0.5, 0.7)
    X = features(A)
    E = A["rawE"]
    ok = (A["dom"] >= 0) & (E >= 1.0) & np.all(np.isfinite(X), axis=1)
    y = A["dom"] == 0
    va = ok & (A["chain"] % 4 == 3)
    tr = ok & ~(A["chain"] % 4 == 3)
    net = load_net(net_path)
    with torch.no_grad():
        u = np.concatenate([to_np(net.logit(torch.tensor(X[i:i + 500000]))) for i in range(0, len(X), 500000)])
    s = 1.0 / (1.0 + np.exp(-u.astype(np.float64)))
    # merge energy bins until each holds MIN_HAD train hadrons
    had = tr & ~y
    edges = [EDGES[0]]
    for e in EDGES[1:]:
        if ((E >= edges[-1]) & (E < e) & had).sum() >= MIN_HAD or e == EDGES[-1]:
            edges.append(e)
    if ((E >= edges[-2]) & had).sum() < MIN_HAD and len(edges) > 2:
        edges.pop(-2)
    centres, sk, ck = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = had & (E >= lo) & (E < hi)
        centres.append(float(np.median(X[m, 0])))
        a, idx = np.unique(np.round(np.quantile(u[m], QS), 4), return_index=True)
        b = np.maximum.accumulate(np.quantile(cnn[m], QS)[idx])
        sk.append(a)
        ck.append(b)
        print(f"bin {lo:g}-{hi:g} GeV: {m.sum()} train hadrons, centre E {np.expm1(centres[-1]):.1f} GeV, knots {len(a)}")
    t = X[:, 0]
    g = numpy_map(u, t, centres, sk, ck)
    # the deployed v1 model is the network and np.interp on these knots (parity 1e-6, train logs)
    v1 = np.interp(s, *v1_knots)
    iv = np.where(va)[0]
    print(f"{nC} chains; validation AUC CNN {auc(cnn[va], y[va].astype(np.float32)):.4f} "
          f"shape {auc(s[va], y[va].astype(np.float32)):.4f}")
    print("validation: hadron mis-ID and EM efficiency per raw-energy bin, CNN / v1 map / energy map")
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = va & (E >= lo) & (E < hi)
        h, e = m & ~y, m & y
        if h.sum() < 50:
            continue
        row = f"  E {lo:>3g}-{hi:<5g} had {h.sum():7d} EM {e.sum():7d} |"
        for th in thresholds:
            row += (f" t{th}: mis-ID {np.mean(cnn[h] >= th):.3f}/{np.mean(v1[h] >= th):.3f}/{np.mean(g[h] >= th):.3f}"
                    f" eff {np.mean(cnn[e] >= th):.2f}/{np.mean(v1[e] >= th):.2f}/{np.mean(g[e] >= th):.2f} |")
        print(row)
    for th in thresholds:
        h, e = va & ~y, va & y
        print(f"  all, t{th}: mis-ID {np.mean(cnn[h] >= th):.4f}/{np.mean(v1[h] >= th):.4f}/{np.mean(g[h] >= th):.4f}"
              f"  eff {np.mean(cnn[e] >= th):.3f}/{np.mean(v1[e] >= th):.3f}/{np.mean(g[e] >= th):.3f}")
    model = EMapped(net, centres, sk, ck).eval()
    torch.onnx.export(model, torch.tensor(X[:4]), out + ".onnx", input_names=["input"], output_names=["em"],
                      dynamic_axes={"input": {0: "n"}, "em": {0: "n"}}, opset_version=17, dynamo=False)
    json.dump({"edges": edges, "centres_log1pE": centres, "logit": [list(map(float, a)) for a in sk],
               "cnn": [list(map(float, b)) for b in ck]}, open(out + "_map.json", "w"))
    # onnxruntime and the torch ONNX export in one process abort on a duplicate protobuf descriptor
    with tempfile.TemporaryDirectory() as d:
        np.save(f"{d}/x.npy", X[iv[:50000]])
        np.save(f"{d}/g.npy", g[iv[:50000]])
        code = ("import sys, numpy as np, onnxruntime as ort; x = np.load(sys.argv[2]); g = np.load(sys.argv[3]); "
                "o = ort.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider']).run(None, {'input': x})[0]; "
                "print(float(np.max(np.abs(o.reshape(-1) - g))))")
        r = subprocess.run([sys.executable, "-I", "-c", code, out + ".onnx", f"{d}/x.npy", f"{d}/g.npy"],
                           capture_output=True, text=True, check=True)
    print(f"onnx vs numpy map: max diff {float(r.stdout.split()[-1]):.2e}")


if __name__ == "__main__":
    main()
