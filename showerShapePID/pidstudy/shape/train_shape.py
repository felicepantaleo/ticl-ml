#!/usr/bin/env python3
"""Shower-shape EM classifier of the CLUE3DHigh tracksters (stage 1 of the PID replacement, pidstudy/REPORT.md):
an MLP on the 16 trackster inputs of the study (ana_hi.py) predicts P(EM), EM = the dominant calo-boundary object is a
photon, an electron or a pi0 (StageTruthProbe). Then the monotone map g of the score s onto the scale of the CNN it
replaces (P(photon) + P(electron) of CNN/patternrecognition/id_v0): g matches the quantiles of the two on the hadron
tracksters, so every threshold t of the chain keeps the hadron mis-ID of the CNN and only the EM efficiency changes.

Rows: probe chains of pidstudy/probe, raw energy >= 1 GeV, tracksters with a simulated deposit. Split by chain: every
fourth chain is validation. Output: <out>.onnx (input "input" [N, 16], output "em" [N]) and <out>_map.json (the knots
of g: s and CNN values at the hadron quantiles 0, 0.005, ..., 1).

usage: train_shape.py <out stem>   (cmsenv python)
"""
import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import uproot

# Root of the pidstudy scripts and reference models (default: this repository).
PIDSTUDY = os.environ.get("PIDSTUDY", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

S = os.path.join(PIDSTUDY, "probe")
torch.manual_seed(20261015)


def to_np(t):
    return np.array(t.detach().tolist(), dtype=np.float32)


def load():
    rows = []
    dirs = sorted(glob.glob(f"{S}/*/0*/done"))
    for ci, d in enumerate(dirs):
        d = os.path.dirname(d)
        st = uproot.open(f"{d}/ntuple.root")["stageProbe/ts"].arrays(["event", "stage", "nLC", "rawE", "domSpecies",
                                                                    "prob"], library="np")
        m = st["stage"] == 0
        st = {k: v[m] for k, v in st.items()}
        hi = uproot.open(f"{d}/ts_flat.root")["hi"].arrays(library="np")
        o1 = np.argsort(st["event"], kind="stable")
        o2 = np.argsort(hi["event"], kind="stable")
        st = {k: v[o1] for k, v in st.items()}
        hi = {k: v[o2] for k, v in hi.items()}
        assert len(st["event"]) == len(hi["event"]) and (st["event"] == hi["event"]).all(), d
        assert (st["nLC"] == hi["nLC"]).all(), d
        n = len(hi["event"])
        r = {"chain": np.full(n, ci), "dom": st["domSpecies"]}
        for k in ("rawE", "rawEmE", "eta", "absZ", "nLC", "eig", "sig", "sigPCA", "axis", "pCNNA"):
            r[k] = np.asarray(hi[k])
        rows.append(r)
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}, len(dirs)


def features(A):
    E = A["rawE"]
    eta = A["eta"]
    eig = np.maximum(A["eig"], 1e-6)
    axisEta = np.arcsinh(A["axis"][:, 2] / np.maximum(np.hypot(A["axis"][:, 0], A["axis"][:, 1]), 1e-6))
    return np.column_stack([np.log1p(E), A["rawEmE"] / np.maximum(E, 1e-6), np.abs(eta), A["absZ"], np.log1p(A["nLC"]),
                            np.log(eig), eig[:, 0] / eig.sum(1), A["sig"], A["sigPCA"],
                            np.abs(axisEta - eta)]).astype(np.float32)


class Net(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))
        self.body = torch.nn.Sequential(torch.nn.Linear(len(mean), 64), torch.nn.SiLU(), torch.nn.Linear(64, 64),
                                        torch.nn.SiLU(), torch.nn.Linear(64, 1))

    def logit(self, x):
        return self.body((x - self.mean) / self.std)[:, 0]

    def forward(self, x):
        return torch.sigmoid(self.logit(x))


class Mapped(torch.nn.Module):
    """The score of net mapped by the piecewise-linear g through the knots (sk, ck): g(s) = c0 + sum of the slope changes
    times relu(s - s_k), clamped to [c0, c_last] and to [0, 1]."""

    def __init__(self, net, sk, ck):
        super().__init__()
        self.net = net
        sk = np.asarray(sk, np.float64)
        ck = np.asarray(ck, np.float64)
        slope = np.diff(ck) / np.maximum(np.diff(sk), 1e-12)
        dslope = np.diff(np.concatenate([[0.0], slope]))
        self.register_buffer("knots", torch.tensor(sk[:-1], dtype=torch.float32))
        self.register_buffer("dslope", torch.tensor(dslope, dtype=torch.float32))
        self.c0, self.c1 = float(ck[0]), float(ck[-1])

    def forward(self, x):
        s = self.net(x).unsqueeze(1)
        g = self.c0 + (torch.relu(s - self.knots) * self.dslope).sum(dim=1)
        return torch.clamp(torch.clamp(g, self.c0, self.c1), 0.0, 1.0)


def auc(s, y):
    o = np.argsort(s)
    y = y[o]
    npos, nneg = y.sum(), (1 - y).sum()
    ranks = np.arange(1, len(y) + 1)
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    out = sys.argv[1]
    A, nC = load()
    X = features(A)
    ok = (A["dom"] >= 0) & (A["rawE"] >= 1.0) & np.all(np.isfinite(X), axis=1)
    y = (A["dom"] == 0).astype(np.float32)
    va_chain = A["chain"] % 4 == 3
    tr, va = ok & ~va_chain, ok & va_chain
    print(f"tracksters {len(y)}, used {ok.sum()} (EM {int(y[ok].sum())}); train {tr.sum()} val {va.sum()} ({nC} chains)")
    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-6
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = Net(mean, std).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    Xt, yt = torch.tensor(X[tr]).to(dev), torch.tensor(y[tr]).to(dev)
    Xv, yv = torch.tensor(X[va]).to(dev), torch.tensor(y[va]).to(dev)
    best, best_state, wait = 1e9, None, 0
    for epoch in range(200):
        net.train()
        perm = torch.randperm(len(Xt), device=dev)
        for b in range(0, len(Xt), 8192):
            idx = perm[b:b + 8192]
            opt.zero_grad()
            loss = bce(net.logit(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            lv = bce(net.logit(Xv), yv).item()
        if lv < best - 1e-5:
            best, best_state, wait = lv, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 10:
                break
    net.load_state_dict(best_state)
    net = net.cpu().eval()
    with torch.no_grad():
        s = to_np(net(torch.tensor(X)))
    cnn = A["pCNNA"][:, 0] + A["pCNNA"][:, 1]
    print(f"epochs {epoch + 1}, best val BCE {best:.5f}")
    print(f"val AUC: CNN {auc(cnn[va], y[va]):.4f}  shape MLP {auc(s[va], y[va]):.4f}")
    # The map g: hadron quantiles of s onto the hadron quantiles of the CNN (all usable rows).
    had = ok & (y == 0)
    qs = np.linspace(0, 1, 201)
    sk = np.quantile(s[had], qs)
    ck = np.quantile(cnn[had], qs)
    sk, idx = np.unique(sk, return_index=True)
    ck = np.maximum.accumulate(ck[idx])
    g = np.interp(s, sk, ck)
    print("validation, per working point of the chain: CNN EM eff / mis-ID, mapped shape MLP EM eff / mis-ID")
    for name, t in (("Cornetto anchor em > 0.6", 0.6), ("supercluster seed >= 0.1", 0.1),
                    ("supercluster emission >= 0.3", 0.3), ("isHadronic P(g)+P(e) >= 0.5", 0.5)):
        em_v, had_v = va & (y == 1), va & (y == 0)
        print(f"  {name}: CNN {np.mean(cnn[em_v] > t):.3f} / {np.mean(cnn[had_v] > t):.4f}   "
              f"shape {np.mean(g[em_v] > t):.3f} / {np.mean(g[had_v] > t):.4f}")
    torch.onnx.export(net, torch.tensor(X[:4]), out + ".onnx", input_names=["input"], output_names=["em"],
                      dynamic_axes={"input": {0: "n"}, "em": {0: "n"}}, opset_version=17, dynamo=False)
    json.dump({"score": [float(v) for v in sk], "cnn": [float(v) for v in ck]}, open(out + "_map.json", "w"))
    # The deployed model: the network and the map g in one graph, as a sum of ramps clamped to [0, 1].
    mapped = Mapped(net, sk, ck).eval()
    with torch.no_grad():
        gm = to_np(mapped(torch.tensor(X[va][:20000])))
    print(f"mapped model vs np.interp: max diff {np.max(np.abs(gm - g[va][:20000])):.2e}")
    torch.onnx.export(mapped, torch.tensor(X[:4]), out + "_mapped.onnx", input_names=["input"], output_names=["em"],
                      dynamic_axes={"input": {0: "n"}, "em": {0: "n"}}, opset_version=17, dynamo=False)
    with tempfile.TemporaryDirectory() as d:
        np.save(f"{d}/x.npy", X[va][:2000])
        np.save(f"{d}/e.npy", s[va][:2000])
        code = ("import numpy as np, onnxruntime as ort; x = np.load('%s/x.npy'); e = np.load('%s/e.npy'); "
                "o = ort.InferenceSession('%s', providers=['CPUExecutionProvider']).run(None, {'input': x})[0]; "
                "print(float(np.max(np.abs(o.reshape(e.shape) - e))))" % (d, d, out + ".onnx"))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        print(f"onnx parity {r.stdout.split()[-1]}; map knots {len(sk)}")


if __name__ == "__main__":
    main()
