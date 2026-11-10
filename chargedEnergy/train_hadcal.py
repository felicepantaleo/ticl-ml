#!/usr/bin/env python3
"""Calibration of the hadronic tracksters that charged hadrons share in the TICLv6 arbitration: a quantile network
predicts the 16, 50 and 84% quantiles of t = (sum of the true energies of the tracks' particles) / (regressed energy)
from calorimeter features only (no track momentum: the track is the other measurement of the average).

Rows: calibProbe/groups of calib/{tt,qcd}/<chain>/ntuple.root with a hadronic trackster and every track matched to a
truth particle with purity >= MIN_PURITY. Split by chain: the chains of the last VAL_FRACTION are the validation set.
Then the emulation on the validation rows: each track's energy from the inverse-variance fit with the calibrated
trackster energy (median) and its error ((q84 - q16) / 2), against the track energy alone, versus the true energy.

usage: train_hadcal.py <out.onnx>    (cmsenv python; CPU)
"""
import glob
import os
import sys

import numpy as np
import torch
import uproot


def to_np(t):
    """Tensor to numpy without the numpy bridge (this torch build has none)."""
    return np.array(t.detach().tolist(), dtype=np.float32)

C = "/data/fpantale/ticlv6retrain/calib"
MIN_PURITY = 0.5
VAL_FRACTION = 0.25
NSIGMA = 3.0
FEATURES = ["log regE", "log rawE", "emFrac", "absEta", "log1p nLC", "sigPCA0", "sigPCA1", "sigPCA2", "nTracks",
            "nJetMembers"]
torch.manual_seed(20261015)
np.random.seed(20261015)


def features(g):
    regE = np.maximum(g["regE"], 1e-3)
    rawE = np.maximum(g["rawE"], 1e-3)
    return np.stack([np.log(regE), np.log(rawE), g["rawEmE"] / rawE, g["absEta"], np.log1p(g["nLC"]),
                     g["sigPCA"][:, 0], g["sigPCA"][:, 1], g["sigPCA"][:, 2], g["nTracks"].astype(np.float32),
                     g["nJetMembers"].astype(np.float32)], axis=1).astype(np.float32)


def load():
    rows = []
    for d in sorted(glob.glob(f"{C}/tt/*/done") + glob.glob(f"{C}/qcd/*/done")):
        sample = d.split("/")[-3]
        chain = int(d.split("/")[-2])
        t = uproot.open(os.path.join(os.path.dirname(d), "ntuple.root"))["calibProbe/groups"].arrays(library="np")
        t["chainKey"] = np.full(len(t["regE"]), (0 if sample == "tt" else 1) * 100000 + chain)
        rows.append(t)
    keys = [k for k in rows[0] if k not in ("trkE", "trkVar", "trkTrueE", "trkPurity", "trkTruePdg", "trkMode")]
    g = {k: np.concatenate([r[k] for r in rows]) for k in keys}
    for k in ("trkE", "trkVar", "trkTrueE", "trkPurity", "trkTruePdg", "trkMode"):
        g[k] = np.concatenate([r[k] for r in rows])
    return g


def onnx_parity(path, x, expected):
    """Max |ONNX Runtime - torch| on x, in a separate process: onnxruntime and torch do not load in one process here."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        np.save(f"{d}/x.npy", x)
        np.save(f"{d}/e.npy", expected)
        code = ("import numpy as np, onnxruntime as ort; x = np.load('%s/x.npy'); e = np.load('%s/e.npy'); "
                "o = ort.InferenceSession('%s', providers=['CPUExecutionProvider']).run(None, {'input': x})[0]; "
                "print(float(np.max(np.abs(o.reshape(e.shape) - e))))" % (d, d, path))
        return float(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                    check=True).stdout.split()[-1])


class Net(torch.nn.Module):
    """Standardised inputs, then the log-quantiles q16 <= q50 <= q84 of t; the output is exp of them."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))
        self.body = torch.nn.Sequential(torch.nn.Linear(len(mean), 64), torch.nn.SiLU(), torch.nn.Linear(64, 64),
                                        torch.nn.SiLU(), torch.nn.Linear(64, 3))

    def logq(self, x):
        o = self.body((x - self.mean) / self.std)
        mid = o[:, 1]
        return torch.stack([mid - torch.nn.functional.softplus(o[:, 0]), mid,
                            mid + torch.nn.functional.softplus(o[:, 2])], dim=1)

    def forward(self, x):
        return torch.exp(self.logq(x))


def pinball(logq, y):
    loss = 0.
    for i, q in enumerate((0.16, 0.5, 0.84)):
        d = y - logq[:, i]
        loss = loss + torch.mean(torch.maximum(q * d, (q - 1) * d))
    return loss


def main():
    out = sys.argv[1]
    g = load()
    n = len(g["regE"])
    good = (g["hadronic"] == 1) & (g["regE"] > 0)
    true_sum = np.zeros(n)
    all_matched = np.ones(n, bool)
    for i in range(n):
        te, pu = g["trkTrueE"][i], g["trkPurity"][i]
        all_matched[i] = len(te) > 0 and np.all(te > 0) and np.all(pu >= MIN_PURITY)
        true_sum[i] = te.sum()
    sel = good & all_matched
    keys = np.unique(g["chainKey"])
    rng = np.random.default_rng(1)
    val_keys = set(rng.choice(keys, int(len(keys) * VAL_FRACTION), replace=False))
    is_val = np.array([k in val_keys for k in g["chainKey"]])
    X = features(g)
    y = np.log(true_sum / g["regE"]).astype(np.float32)
    tr, va = sel & ~is_val, sel & is_val
    print(f"groups {n}, hadronic {good.sum()}, all tracks matched (purity >= {MIN_PURITY}) {sel.sum()}; "
          f"train {tr.sum()} val {va.sum()} ({len(keys)} chains, {len(val_keys)} val)")
    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-6
    net = Net(mean, std)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    Xt, yt = torch.tensor(X[tr]), torch.tensor(y[tr])
    Xv, yv = torch.tensor(X[va]), torch.tensor(y[va])
    best, best_state, wait = 1e9, None, 0
    for epoch in range(300):
        net.train()
        perm = torch.randperm(len(Xt))
        for b in range(0, len(Xt), 1024):
            idx = perm[b:b + 1024]
            opt.zero_grad()
            loss = pinball(net.logq(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            lv = pinball(net.logq(Xv), yv).item()
        if lv < best - 1e-5:
            best, best_state, wait = lv, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 20:
                break
    net.load_state_dict(best_state)
    print(f"epochs {epoch + 1}, best val pinball {best:.5f}")
    net.eval()
    with torch.no_grad():
        q = to_np(net(torch.tensor(X)))
    t = true_sum / g["regE"]
    cov = np.mean((t[va] >= q[va, 0]) & (t[va] <= q[va, 2]))
    print(f"val coverage of [q16, q84]: {cov:.3f} (0.68 expected); median of t/q50: {np.median(t[va] / q[va, 1]):.3f}")
    for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)):
        m = va & (g["regE"] >= lo) & (g["regE"] < hi)
        if m.sum() < 20:
            continue
        print(f"  regE {lo}-{hi}: n {m.sum()}  median t {np.median(t[m]):.3f}  median q50 {np.median(q[m, 1]):.3f}  "
              f"rel width (q84-q16)/2q50 {np.median((q[m, 2] - q[m, 0]) / 2 / q[m, 1]):.3f}  "
              f"coverage {np.mean((t[m] >= q[m, 0]) & (t[m] <= q[m, 2])):.3f}")

    # Emulation on the validation rows: per track, the track energy alone, the uncalibrated average (2.32/sqrt(E), the
    # regressed energy) and the calibrated average, against the true energy of its particle.
    def fit(i, eTs, sigTs):
        e, v = g["trkE"][i], g["trkVar"][i]
        diff = eTs - e.sum()
        vt = v.sum() + sigTs ** 2
        if not abs(diff) < NSIGMA * np.sqrt(vt):
            return e
        new = e + v / vt * diff
        return e if np.any(new <= 0.13957) else new
    res = {"track": [], "average 2.32/sqrtE": [], "calibrated average": []}
    pt_true = []
    for i in np.where(va)[0]:
        eSum = g["trkE"][i].sum()
        res["track"].append(g["trkE"][i] / g["trkTrueE"][i])
        res["average 2.32/sqrtE"].append(fit(i, g["regE"][i], 2.32 * np.sqrt(eSum)) / g["trkTrueE"][i])
        eCal = g["regE"][i] * q[i, 1]
        res["calibrated average"].append(fit(i, eCal, g["regE"][i] * (q[i, 2] - q[i, 0]) / 2) / g["trkTrueE"][i])
        pt_true.append(g["trkTrueE"][i])
    pt_true = np.concatenate(pt_true)
    print("emulation on the val tracks: median and sigma68/median of E/E_true by true energy")
    for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)):
        m = (pt_true >= lo) & (pt_true < hi)
        line = f"  E_true {lo}-{hi}: n {m.sum()}"
        for k, v in res.items():
            r = np.concatenate(v)[m]
            q16, q50, q84 = np.quantile(r, [0.16, 0.5, 0.84])
            line += f" | {k}: {q50:.4f} {0.5 * (q84 - q16) / q50:.4f}"
        print(line)
    # Variants of the combination on the validation tracks: track error scale (the probe stores scale 2) and the
    # compatibility cut, against the track alone; single-track and multi-track groups apart.
    def emulate(scale2, nsigma, multi):
        out_r = []
        for i in np.where(va)[0]:
            ntrk = len(g["trkE"][i])
            if (ntrk > 1) != multi:
                continue
            e, v = g["trkE"][i], g["trkVar"][i] * (scale2 / 2.0) ** 2
            eCal, sig = g["regE"][i] * q[i, 1], g["regE"][i] * (q[i, 2] - q[i, 0]) / 2
            diff = eCal - e.sum()
            vt = v.sum() + sig ** 2
            new = e + v / vt * diff if abs(diff) < nsigma * np.sqrt(vt) else e
            if np.any(new <= 0.13957):
                new = e
            out_r.append(new / g["trkTrueE"][i])
        r = np.concatenate(out_r)
        q16, q50, q84 = np.quantile(r, [0.16, 0.5, 0.84])
        return f"{q50:.4f} {0.5 * (q84 - q16) / q50:.4f} tails>10% {np.mean(np.abs(r - 1) > 0.1):.4f}"
    for multi in (False, True):
        print(f"variants, {'multi' if multi else 'single'}-track groups: track alone "
              + emulate(1e-9, 0.0, multi))
        for scale2 in (1.0, 2.0):
            for nsigma in (1.0, 2.0, 3.0):
                print(f"  track error scale {scale2}, compatibility {nsigma} sigma: " + emulate(scale2, nsigma, multi))
    np.savez(out.replace(".onnx", "_val.npz"), q=q[va], regE=g["regE"][va], X=X[va])
    torch.onnx.export(net, torch.tensor(X[:4]), out, input_names=["input"], output_names=["quantiles"],
                      dynamic_axes={"input": {0: "n"}, "quantiles": {0: "n"}}, opset_version=17, dynamo=False)
    print(f"onnx parity: max |onnx - torch| {onnx_parity(out, X[va][:1000], q[va][:1000]):.2e}; features {FEATURES}")


if __name__ == "__main__":
    main()
