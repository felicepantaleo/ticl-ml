#!/usr/bin/env python3
"""Deployed ONNX of the mixture-aware PID (pidstudy/mix): input "input" [N, 22] = the 16 trackster quantities of the
shape PID and the 6 raw layer-cluster profile values (largest share, core shares at 2 and 5 cm, transverse rms, min |z|,
|z| extent; ticl::layerClusterProfile); output "pid" [N, 8] in Trackster::ParticleType order.

The network gives the species fractions (photon, electron, pi0, charged hadron, neutral hadron, muon, other). Their EM
sum s = photon + electron + pi0 is mapped per raw energy onto the CNN scale (photon + electron of the CNN it replaces) by
matching the hadron quantiles (hadron: truth EM fraction < 0.5) in raw-energy bins, linear in log(1 + E) between the bin
centres (pidstudy/shape/emap.py on the logit of s). The EM slots are scaled by g/s, the other slots by (1 - g)/(1 - s),
so every threshold on photon + electron keeps the CNN hadron mis-identification in every energy bin.

usage: export_mix.py <hi|fin> <model stem>   (reads <stem>.pt, writes <stem>_mapped.onnx and <stem>_map.json)
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

# Root of the pidstudy scripts and reference models (default: this repository).
PIDSTUDY = os.environ.get("PIDSTUDY", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, os.path.join(PIDSTUDY, "mix"))
sys.path.insert(0, os.path.join(PIDSTUDY, "shape"))
from train_mix import Net, inputs, load, to_np  # noqa: E402
from emap import EDGES, EMapped, numpy_map  # noqa: E402

EPS = 1e-6
QS = np.linspace(0, 1, 201)
MIN_HAD = 1000


class Profile(torch.nn.Module):
    """The input transform of train_mix.inputs for the 6 raw profile values."""

    def forward(self, x):
        p = x[:, 16:]
        q = torch.stack([p[:, 0], p[:, 1], p[:, 2], torch.log1p(torch.clamp(p[:, 3], min=0.0)), p[:, 4] / 100.0,
                         p[:, 5] / 100.0], dim=1)
        return torch.cat([x[:, :16], q], dim=1)


class EMLogit(torch.nn.Module):
    """logit of the EM sum of the fractions: the score that the energy map takes."""

    def __init__(self, net):
        super().__init__()
        self.net, self.prof = net, Profile()

    def fractions(self, x):
        return self.net(self.prof(x))

    def logit(self, x):
        f = self.fractions(x)
        s = torch.clamp(f[:, 0] + f[:, 1] + f[:, 2], EPS, 1 - EPS)
        return torch.log(s) - torch.log1p(-s)


class Deployed(torch.nn.Module):
    def __init__(self, em, centres, sk, ck):
        super().__init__()
        self.em = em
        self.mapped = EMapped(em, centres, sk, ck)

    def forward(self, x):
        f = self.em.fractions(x)
        s = torch.clamp(f[:, 0] + f[:, 1] + f[:, 2], EPS, 1 - EPS)
        g = self.mapped(x)
        a, b = g / s, (1 - g) / (1 - s)
        z = torch.zeros_like(s)
        # ParticleType: photon, electron, muon, neutral_pion, charged_hadron, neutral_hadron, ambiguous, unknown
        return torch.stack([f[:, 0] * a, f[:, 1] * a, f[:, 5] * b, f[:, 2] * a, f[:, 3] * b, f[:, 4] * b, z, f[:, 6] * b],
                           dim=1)


def main():
    level, stem = sys.argv[1], sys.argv[2]
    A, nC = load(level)
    X = inputs(A)
    Xraw = np.column_stack([X[:, :16], A["lcProf"]]).astype(np.float32)
    F = A["frac"]
    E = A["rawE"]
    ok = (E >= 1.0) & (A["dom"] >= 0) & np.all(np.isfinite(X), axis=1) & (np.abs(F.sum(1) - 1) < 1e-3)
    tr = ok & ~(A["chain"] % 4 == 3)
    va = ok & (A["chain"] % 4 == 3)
    sd = torch.load(stem + ".pt")
    net = Net(np.array(sd["mean"].tolist(), np.float32), np.array(sd["std"].tolist(), np.float32))
    net.load_state_dict(sd)
    net.eval()
    em = EMLogit(net).eval()
    with torch.no_grad():
        u = np.concatenate([to_np(em.logit(torch.tensor(Xraw[i:i + 500000]))) for i in range(0, len(Xraw), 500000)])
    cnn = A["prob"][:, 0] + A["prob"][:, 1]
    had = tr & (F[:, 0] + F[:, 1] + F[:, 2] < 0.5)
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
        sk.append(a)
        ck.append(np.maximum.accumulate(np.quantile(cnn[m], QS)[idx]))
        print(f"bin {lo:g}-{hi:g} GeV: {m.sum()} train hadrons, centre {np.expm1(centres[-1]):.1f} GeV, knots {len(a)}")
    g = numpy_map(u, X[:, 0], centres, sk, ck)
    femT = F[:, 0] + F[:, 1] + F[:, 2]
    print("validation: hadron mis-ID and EM efficiency (truth EM fraction >= 0.5) per bin, CNN / mapped mixture")
    thr = (0.1, 0.3, 0.5, 0.6) if level == "hi" else (0.3, 0.5, 0.7)
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = va & (E >= lo) & (E < hi)
        h, e = m & (femT < 0.5), m & (femT >= 0.5)
        if h.sum() < 50:
            continue
        print(f"  E {lo:>3g}-{hi:<5g} had {h.sum():7d} EM {e.sum():7d} |" + "".join(
            f" t{t}: {np.mean(cnn[h] >= t):.3f}/{np.mean(g[h] >= t):.3f} eff {np.mean(cnn[e] >= t):.2f}/{np.mean(g[e] >= t):.2f} |"
            for t in thr))
    model = Deployed(em, centres, sk, ck).eval()
    torch.onnx.export(model, torch.tensor(Xraw[:4]), stem + "_mapped.onnx", input_names=["input"], output_names=["pid"],
                      dynamic_axes={"input": {0: "n"}, "pid": {0: "n"}}, opset_version=17, dynamo=False)
    json.dump({"edges": edges, "centres_log1pE": centres, "logit": [list(map(float, a)) for a in sk],
               "cnn": [list(map(float, b)) for b in ck]}, open(stem + "_map.json", "w"))
    iv = np.where(va)[0][:50000]
    with torch.no_grad():
        ref = to_np(model(torch.tensor(Xraw[iv])))
    print(f"torch mapped EM sum vs numpy map: max diff {np.max(np.abs(ref[:, 0] + ref[:, 1] + ref[:, 3] - g[iv])):.2e}; "
          f"slot sum max |1 - sum| {np.max(np.abs(ref.sum(1) - 1)):.2e}")
    with tempfile.TemporaryDirectory() as d:
        np.save(f"{d}/x.npy", Xraw[iv])
        np.save(f"{d}/r.npy", ref)
        code = ("import sys, numpy as np, onnxruntime as ort; x = np.load(sys.argv[2]); r = np.load(sys.argv[3]); "
                "o = ort.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider']).run(None, {'input': x})[0]; "
                "print(float(np.max(np.abs(o - r))))")
        r = subprocess.run([sys.executable, "-I", "-c", code, stem + "_mapped.onnx", f"{d}/x.npy", f"{d}/r.npy"],
                           capture_output=True, text=True, check=True)
    print(f"onnx parity {float(r.stdout.split()[-1]):.2e}")
    np.save(stem + "_parity_x.npy", Xraw[iv][:2000])
    np.save(stem + "_parity_pid.npy", ref[:2000])


if __name__ == "__main__":
    main()
