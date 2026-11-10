#!/usr/bin/env python3
"""Shower-shape EM classifier of the final tracksters and footprints of the TICLv6 interpretation stage (stage 2 of the
PID replacement, pidstudy/REPORT.md Section 7): an MLP on the 16 trackster inputs of TracksterInferenceByShowerShape
predicts P(EM), EM = the dominant calo-boundary object is a photon, an electron or a pi0 (StageTruthProbe stage 4).
Then the monotone map g of the score onto the scale of the linking CNN that it replaces (P(photon) + P(electron) of
CNN/linking/id_v0, the isHadronic sum): g matches the quantiles of the two on the hadron tracksters, so every threshold
of the interpretation stage keeps the hadron mis-ID of the linking CNN and only the EM efficiency changes.

Rows: stage-2 probe chains (stage2/probe, train chains, code with the stage-1 shape PID), final tracksters with raw
energy >= 1 GeV and a simulated deposit. Split by chain: every fourth chain is validation.
Output: <out>.onnx (score), <out>_mapped.onnx (input "input" [N, 16], output "em" [N], the deployed model) and
<out>_map.json (the knots of g).

usage: train_final.py <out stem>   (cmsenv python)
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
from train_shape import Mapped, Net, auc, features, to_np  # noqa: E402  same inputs and network as stage 1

S = os.path.join(PIDSTUDY, "stage2/probe")
torch.manual_seed(20261015)


def load():
    rows = []
    dirs = sorted(os.path.dirname(d) for d in glob.glob(f"{S}/*/0*/done") if os.path.exists(os.path.dirname(d) + "/ts_flat.root"))
    for ci, d in enumerate(dirs):
        st = uproot.open(f"{d}/ntuple.root")["stageProbe/ts"].arrays(["event", "stage", "nLC", "rawE", "domSpecies"], library="np")
        m = st["stage"] == 4
        st = {k: v[m] for k, v in st.items()}
        fi = uproot.open(f"{d}/ts_flat.root")["fin"].arrays(library="np")
        o1 = np.argsort(st["event"], kind="stable")
        o2 = np.argsort(fi["event"], kind="stable")
        st = {k: v[o1] for k, v in st.items()}
        fi = {k: v[o2] for k, v in fi.items()}
        assert len(st["event"]) == len(fi["event"]) and (st["event"] == fi["event"]).all(), d
        assert (st["nLC"] == fi["nLC"]).all(), d
        n = len(fi["event"])
        r = {"chain": np.full(n, ci), "dom": st["domSpecies"], "sample": np.full(n, d.split("/")[-2])}
        for k in ("rawE", "rawEmE", "eta", "absZ", "nLC", "eig", "sig", "sigPCA", "axis", "pid"):
            r[k] = np.asarray(fi[k])
        rows.append(r)
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}, len(dirs)


def main():
    out = sys.argv[1]
    A, nC = load()
    X = features(A)
    ok = (A["dom"] >= 0) & (A["rawE"] >= 1.0) & np.all(np.isfinite(X), axis=1)
    y = (A["dom"] == 0).astype(np.float32)
    va_chain = A["chain"] % 4 == 3
    tr, va = ok & ~va_chain, ok & va_chain
    print(f"final tracksters {len(y)}, used {ok.sum()} (EM {int(y[ok].sum())}); train {tr.sum()} val {va.sum()} ({nC} chains)")
    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-6
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = Net(mean, std).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    Xt, yt = torch.tensor(X[tr]).to(dev), torch.tensor(y[tr]).to(dev)
    Xv, yv = torch.tensor(X[va]).to(dev), torch.tensor(y[va]).to(dev)
    best, best_state, wait = 1e9, None, 0
    for epoch in range(300):
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
    link = A["pid"][:, 0] + A["pid"][:, 1]
    print(f"epochs {epoch + 1}, best val BCE {best:.5f}")
    print(f"val AUC: linking CNN {auc(link[va], y[va]):.4f}  shape MLP {auc(s[va], y[va]):.4f}")
    for lo, hi in ((1, 5), (5, 20), (20, 50), (50, 1e9)):
        m = va & (A["rawE"] >= lo) & (A["rawE"] < hi)
        print(f"  E {lo}-{hi:g}: n {m.sum()}  linking CNN {auc(link[m], y[m]):.4f}  shape MLP {auc(s[m], y[m]):.4f}")
    for smp in np.unique(A["sample"]):
        m = va & (A["sample"] == smp)
        print(f"  sample {smp}: n {m.sum()}  linking CNN {auc(link[m], y[m]):.4f}  shape MLP {auc(s[m], y[m]):.4f}")
    # The map g: hadron quantiles of s onto the hadron quantiles of the linking CNN (train rows).
    had = tr & (y == 0)
    qs = np.linspace(0, 1, 201)
    sk = np.quantile(s[had], qs)
    ck = np.quantile(link[had], qs)
    sk, idx = np.unique(sk, return_index=True)
    ck = np.maximum.accumulate(ck[idx])
    g = np.interp(s, sk, ck)
    print("validation at the thresholds of the interpretation stage: linking CNN EM eff / mis-ID, mapped shape MLP")
    for t in (0.3, 0.5, 0.7):
        em_v, had_v = va & (y == 1), va & (y == 0)
        print(f"  P(g)+P(e) >= {t}: linking CNN {np.mean(link[em_v] >= t):.3f} / {np.mean(link[had_v] >= t):.4f}   "
              f"shape {np.mean(g[em_v] >= t):.3f} / {np.mean(g[had_v] >= t):.4f}")
    torch.onnx.export(net, torch.tensor(X[:4]), out + ".onnx", input_names=["input"], output_names=["em"],
                      dynamic_axes={"input": {0: "n"}, "em": {0: "n"}}, opset_version=17, dynamo=False)
    json.dump({"score": [float(v) for v in sk], "linkCNN": [float(v) for v in ck]}, open(out + "_map.json", "w"))
    mapped = Mapped(net, sk, ck).eval()
    with torch.no_grad():
        gm = to_np(mapped(torch.tensor(X[va][:20000])))
    print(f"mapped model vs np.interp: max diff {np.max(np.abs(gm - g[va][:20000])):.2e}")
    torch.onnx.export(mapped, torch.tensor(X[:4]), out + "_mapped.onnx", input_names=["input"], output_names=["em"],
                      dynamic_axes={"input": {0: "n"}, "em": {0: "n"}}, opset_version=17, dynamo=False)
    with tempfile.TemporaryDirectory() as d:
        np.save(f"{d}/x.npy", X[va][:2000])
        np.save(f"{d}/e.npy", g[va][:2000])
        code = ("import numpy as np, onnxruntime as ort; x = np.load('%s/x.npy'); e = np.load('%s/e.npy'); "
                "o = ort.InferenceSession('%s', providers=['CPUExecutionProvider']).run(None, {'input': x})[0]; "
                "print(float(np.max(np.abs(o.reshape(e.shape) - e))))" % (d, d, out + "_mapped.onnx"))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        print(f"onnx parity of the mapped model {r.stdout.split()[-1]}; map knots {len(sk)}")


if __name__ == "__main__":
    main()
