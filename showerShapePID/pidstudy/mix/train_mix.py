#!/usr/bin/env python3
"""Mixture-aware shower-shape PID (pidstudy/mix): one model per level (hi: CLUE3DHigh tracksters, stage 0; fin: final
tracksters of the interpretation stage, stage 4) predicts the truth deposit fractions of a trackster by species
(photon, electron, pi0, charged hadron, neutral hadron, muon, other): a softmax trained on the soft labels of
StageTruthProbe (frac), so the EM fraction is p_gamma + p_e + p_pi0.

Inputs (22): the 16 trackster quantities of the shape PID (train_shape.features) and the 6 layer-cluster profile
features of the neutral share model (StageTruthProbe lcProf: largest share, core shares at 2 and 5 cm, transverse rms,
min |z|, |z| extent). Rows: probe chains of pidstudy/mix (CNN PIDs, the portA reconstruction), raw energy >= 1 GeV,
tracksters with a truth deposit; every fourth chain is validation. Weights: the same total weight per raw-energy bin.

Evaluation on the validation rows, per raw-energy bin, for the CNN (prob slots photon + electron + pi0), the v1 shape
model, the energy-map shape model and this model: the energy-weighted mean |EM fraction error|, and with EM = an EM
truth fraction >= 0.5, the EM efficiency at the hadron mis-ID of the CNN at the code thresholds.

usage: train_mix.py <hi|fin> <out stem>   (cmsenv python); output <out>.pt, <out>.onnx (input "input" [N, 22], output
"fractions" [N, 7]) and <out>_eval.json
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

sys.path.insert(0, os.path.join(PIDSTUDY, "shape"))
from train_shape import features  # noqa: E402

X_DIR = os.environ["PROBE_DIR"]
EDGES = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100, 1e9]
torch.manual_seed(20261016)


def to_np(t):
    return np.array(t.detach().tolist(), dtype=np.float32)


def load(level):
    stage, tree = (0, "hi") if level == "hi" else (4, "fin")
    rows = []
    dirs = sorted(os.path.dirname(d) for d in glob.glob(f"{X_DIR}/*/0*/done") if os.path.exists(os.path.dirname(d) + "/ts_flat.root"))
    for ci, d in enumerate(dirs):
        st = uproot.open(f"{d}/ntuple.root")["stageProbe/ts"].arrays(
            ["event", "stage", "nLC", "rawE", "domSpecies", "frac", "lcProf", "prob"], library="np")
        m = st["stage"] == stage
        st = {k: v[m] for k, v in st.items()}
        fl = uproot.open(f"{d}/ts_flat.root")[tree].arrays(library="np")
        o1, o2 = np.argsort(st["event"], kind="stable"), np.argsort(fl["event"], kind="stable")
        st = {k: v[o1] for k, v in st.items()}
        fl = {k: v[o2] for k, v in fl.items()}
        assert len(st["event"]) == len(fl["event"]) and (st["event"] == fl["event"]).all() and (st["nLC"] == fl["nLC"]).all(), d
        n = len(fl["event"])
        r = {"chain": np.full(n, ci), "sample": np.full(n, d.split("/")[-2]), "dom": st["domSpecies"],
             "frac": np.asarray(st["frac"]), "lcProf": np.asarray(st["lcProf"]), "prob": np.asarray(st["prob"])}
        for k in ("rawE", "rawEmE", "eta", "absZ", "nLC", "eig", "sig", "sigPCA", "axis"):
            r[k] = np.asarray(fl[k])
        rows.append(r)
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}, len(dirs)


def inputs(A):
    p = A["lcProf"]
    extra = np.column_stack([p[:, 0], p[:, 1], p[:, 2], np.log1p(np.maximum(p[:, 3], 0)), p[:, 4] / 100.0, p[:, 5] / 100.0])
    return np.column_stack([features(A), extra]).astype(np.float32)


class Net(torch.nn.Module):
    def __init__(self, mean, std, n_out=7):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))
        self.body = torch.nn.Sequential(torch.nn.Linear(len(mean), 128), torch.nn.SiLU(), torch.nn.Linear(128, 128),
                                        torch.nn.SiLU(), torch.nn.Linear(128, 64), torch.nn.SiLU(), torch.nn.Linear(64, n_out))

    def logits(self, x):
        return self.body((x - self.mean) / self.std)

    def forward(self, x):
        return torch.softmax(self.logits(x), dim=1)


def shape_scores(A, level):
    """EM scores of the v1 shape model and of the energy-map model on the same rows (16 inputs)."""
    sys.path.insert(0, os.path.join(PIDSTUDY, "shape"))
    from emap import load_net, numpy_map
    X16 = features(A)
    if level == "hi":
        net = load_net(os.path.join(PIDSTUDY, "shape/shapePID_v1.onnx"))
        v1 = json.load(open(os.path.join(PIDSTUDY, "shape/shapePID_v1_map.json")))
        v1k = (v1["score"], v1["cnn"])
        em = json.load(open(os.path.join(PIDSTUDY, "emap/shapePID_v2_map.json")))
    else:
        net = load_net(os.path.join(PIDSTUDY, "stage2/model/shapePIDFinal_v1.onnx"))
        v1 = json.load(open(os.path.join(PIDSTUDY, "stage2/model/shapePIDFinal_v1_map.json")))
        v1k = (v1["score"], v1["linkCNN"])
        em = json.load(open(os.path.join(PIDSTUDY, "emap/shapePIDFinal_v2_map.json")))
    with torch.no_grad():
        u = np.concatenate([to_np(net.logit(torch.tensor(X16[i:i + 500000]))) for i in range(0, len(X16), 500000)])
    s = 1.0 / (1.0 + np.exp(-u.astype(np.float64)))
    return np.interp(s, *v1k), numpy_map(u, X16[:, 0], em["centres_log1pE"], [np.array(a) for a in em["logit"]],
                                         [np.array(b) for b in em["cnn"]])


def main():
    level, out = sys.argv[1], sys.argv[2]
    A, nC = load(level)
    X = inputs(A)
    F = A["frac"].astype(np.float32)
    E = A["rawE"]
    ok = (E >= 1.0) & (A["dom"] >= 0) & np.all(np.isfinite(X), axis=1) & (np.abs(F.sum(1) - 1) < 1e-3)
    va = ok & (A["chain"] % 4 == 3)
    tr = ok & ~(A["chain"] % 4 == 3)
    b = np.clip(np.digitize(E, EDGES) - 1, 0, len(EDGES) - 2)
    w = np.zeros(len(E), np.float32)
    for i in range(len(EDGES) - 1):
        m = tr & (b == i)
        if m.sum():
            w[m] = 1.0 / m.sum()
    w[tr] *= tr.sum() / w[tr].sum()
    print(f"{level}: {nC} chains, rows {len(E)}, train {tr.sum()}, val {va.sum()}; train rows per bin "
          + " ".join(str(int((tr & (b == i)).sum())) for i in range(len(EDGES) - 1)), flush=True)
    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-6
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = Net(mean, std).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    Xt, Ft, Wt = torch.tensor(X[tr]).to(dev), torch.tensor(F[tr]).to(dev), torch.tensor(w[tr]).to(dev)
    Xv, Fv = torch.tensor(X[va]).to(dev), torch.tensor(F[va]).to(dev)
    wv = np.zeros(va.sum(), np.float32)
    for i in range(len(EDGES) - 1):
        m = b[va] == i
        if m.sum():
            wv[m] = 1.0 / m.sum()
    Wv = torch.tensor(wv / wv.sum()).to(dev)

    def loss(logits, f, wt):
        return (-(f * torch.log_softmax(logits, dim=1)).sum(1) * wt).sum() / wt.sum()

    best, best_state, wait = 1e9, None, 0
    for epoch in range(300):
        net.train()
        perm = torch.randperm(len(Xt), device=dev)
        for k in range(0, len(Xt), 8192):
            idx = perm[k:k + 8192]
            opt.zero_grad()
            l = loss(net.logits(Xt[idx]), Ft[idx], Wt[idx])
            l.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            lv = loss(net.logits(Xv), Fv, Wv).item()
        sched.step(lv)
        if lv < best - 1e-5:
            best, best_state, wait = lv, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 12:
                break
    net.load_state_dict(best_state)
    net = net.cpu().eval()
    print(f"epochs {epoch + 1}, best val soft cross entropy {best:.5f}", flush=True)
    torch.save(net.state_dict(), out + ".pt")
    with torch.no_grad():
        P = np.concatenate([to_np(net(torch.tensor(X[i:i + 500000]))) for i in range(0, len(X), 500000)])
    femT = F[:, 0] + F[:, 1] + F[:, 2]
    scores = {"CNN": A["prob"][:, 0] + A["prob"][:, 1] + A["prob"][:, 3], "mix": P[:, 0] + P[:, 1] + P[:, 2]}
    scores["shape v1"], scores["shape emap"] = shape_scores(A, level)
    thr = (0.1, 0.3, 0.5, 0.6) if level == "hi" else (0.3, 0.5, 0.7)
    ev = {"level": level, "chains": nC, "edges": EDGES, "bins": []}
    print("validation per raw-energy bin: energy-weighted mean |EM fraction error| and, EM = truth EM fraction >= 0.5, "
          "EM efficiency at the CNN hadron mis-ID of each threshold (scores mapped by hadron quantiles in the bin)")
    for i in range(len(EDGES) - 1):
        m = va & (b == i)
        if m.sum() < 200:
            continue
        em, had = m & (femT >= 0.5), m & (femT < 0.5)
        row = {"lo": EDGES[i], "hi": EDGES[i + 1], "n": int(m.sum()), "nEM": int(em.sum())}
        txt = f"  E {EDGES[i]:>3g}-{EDGES[i + 1]:<5g} n {m.sum():7d} EM {em.sum():6d} |"
        for name, s in scores.items():
            err = float(np.sum(np.abs(s[m] - femT[m]) * E[m]) / np.sum(E[m]))
            effs = []
            for t in thr:
                mis = np.mean(scores["CNN"][had] >= t)
                cut = np.quantile(s[had], 1 - mis) if mis > 0 else np.inf
                effs.append(float(np.mean(s[em] > cut)) if em.sum() else float("nan"))
            row[name] = {"emFracErr": err, "effAtCNNMisID": effs}
            txt += f" {name} {err:.3f} " + "/".join(f"{e:.2f}" for e in effs) + " |"
        ev["bins"].append(row)
        print(txt, flush=True)
    for name, s in scores.items():
        print(f"  all, energy-weighted |EM fraction error| {name}: {np.sum(np.abs(s[va] - femT[va]) * E[va]) / np.sum(E[va]):.4f}")
    json.dump(ev, open(out + "_eval.json", "w"), indent=1)
    torch.onnx.export(net, torch.tensor(X[:4]), out + ".onnx", input_names=["input"], output_names=["fractions"],
                      dynamic_axes={"input": {0: "n"}, "fractions": {0: "n"}}, opset_version=17, dynamo=False)
    with tempfile.TemporaryDirectory() as d:
        np.save(f"{d}/x.npy", X[va][:5000])
        np.save(f"{d}/p.npy", P[va][:5000])
        code = ("import sys, numpy as np, onnxruntime as ort; x = np.load(sys.argv[2]); p = np.load(sys.argv[3]); "
                "o = ort.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider']).run(None, {'input': x})[0]; "
                "print(float(np.max(np.abs(o - p))))")
        r = subprocess.run([sys.executable, "-I", "-c", code, out + ".onnx", f"{d}/x.npy", f"{d}/p.npy"],
                           capture_output=True, text=True, check=True)
    print(f"onnx parity {float(r.stdout.split()[-1]):.2e}")


if __name__ == "__main__":
    main()
