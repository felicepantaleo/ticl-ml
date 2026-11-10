# D2 hypothesis-model training: the network and recipe of train_c2.py (64, 64 SiLU, AdamW 2e-3, early stopping on es)
# with two options that act on the weight levels of the types that compete for a track.
#   SIGW   weight of the signal rows in the loss (fit and es), 1 = the pilot recipe.
#   PLATT  calibration fit on es and put into the exported model: global (one a, b), type_all (a, b per hypothesis
#          type, all es rows of the type), type_sig (a, b per type, signal es rows of the type).
#   PAIRW  weight of the pairwise term: for two hypotheses on the same track where only the first has the type of the
#          track particle (prepare_pairs.py, <tables dir>/pairs_{fit,es}.npz), softplus(logit_wrong - logit_right).
#   PAIRSEL  all (every pair) or lepsig (the pairs of true electron and muon tracks and of signal tracks).
#   ESMODE  sum (early stopping on the pointwise plus the pairwise es loss) or point (on the pointwise es loss).
#   PAIRFILE  pairs (same-track pairs, default), pairs_conf (conflict pairs, prepare_cpairs.py) or pairs_both.
#   PAIRUP  weights of pairs by type, "r:w:f,...": the pairs with right type r and wrong type w (* any) count f times.
# The exported ONNX gives the calibrated weight: the producer runs it with arbitrationPlattA 1 and arbitrationPlattB 0.
# usage: train_d2.py <tables dir> <tag>; env SIGW, PLATT, PAIRW, PAIRSEL, ESMODE, SEED, NFEAT (32); output
# <tables dir>/arb_<tag>.{pt,onnx,json}
import json, os, sys
import numpy as np, torch
from sklearn.linear_model import LogisticRegression

T, tag = sys.argv[1], sys.argv[2]
nfeat = int(os.environ.get("NFEAT", 32))
sigw = float(os.environ.get("SIGW", 1))
platt = os.environ.get("PLATT", "global")
pairw = float(os.environ.get("PAIRW", 0))
pairsel = os.environ.get("PAIRSEL", "all")
esmode = os.environ.get("ESMODE", "sum")
pairfile = os.environ.get("PAIRFILE", "pairs")
pairup = os.environ.get("PAIRUP", "")
assert esmode in ("sum", "point")
seed = int(os.environ.get("SEED", 1))
torch.manual_seed(seed); np.random.seed(seed)
tabs = {s: np.load(f"{T}/{s}.npz") for s in ("fit", "es")}
Xf, yf, sf = tabs["fit"]["X"][:, :nfeat].copy(), tabs["fit"]["y"].astype(np.float32), tabs["fit"]["signal"].astype(bool)
Xe, ye, se = tabs["es"]["X"][:, :nfeat].copy(), tabs["es"]["y"].astype(np.float32), tabs["es"]["signal"].astype(bool)
te = tabs["es"]["type"]
assert not set(np.unique(tabs["fit"]["chain"])) & set(np.unique(tabs["es"]["chain"])), "chains shared by fit and es"
NTYPE = 7  # one-hot slots 0-6 of the model row, in the order of Hypothesis::Type
assert (np.abs(Xe[:, :NTYPE].sum(1) - 1) < 1e-6).all(), "every model row has one type"
mu, sd = Xf.mean(0), Xf.std(0)
sd[sd == 0] = 1.
hidden = [64, 64]


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mu", torch.tensor(mu)); self.register_buffer("sd", torch.tensor(sd))
        layers, n = [], Xf.shape[1]
        for h in hidden:
            layers += [torch.nn.Linear(n, h), torch.nn.SiLU()]; n = h
        layers.append(torch.nn.Linear(n, 1))
        self.f = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.f((x - self.mu) / self.sd).squeeze(-1)


class Calibrated(torch.nn.Module):
    # Weight = a[type] logit + b[type], with the type from the one-hot slots of the row.
    def __init__(self, net, a, b):
        super().__init__()
        self.net = net
        self.register_buffer("a", torch.tensor(a, dtype=torch.float32)); self.register_buffer("b", torch.tensor(b, dtype=torch.float32))

    def forward(self, x):
        t = x[:, :NTYPE]
        return (t @ self.a) * self.net(x) + t @ self.b


dev = "cuda"
# DATADEV=cpu keeps the feature rows of the fit and es sets in host memory; each batch moves to the GPU.
ddev = os.environ.get("DATADEV", dev)
net = Net().to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
bce = torch.nn.BCEWithLogitsLoss(reduction="none")
Xt, yt = torch.tensor(Xf, device=ddev), torch.tensor(yf, device=dev)
wt = torch.tensor(np.where(sf, sigw, 1.).astype(np.float32), device=dev)
Xet, yet = torch.tensor(Xe, device=ddev), torch.tensor(ye, device=dev)
wet = torch.tensor(np.where(se, sigw, 1.).astype(np.float32), device=dev)
batch, patience = 4096, 10
if pairw > 0:
    pairs = {s: np.load(f"{T}/{pairfile}_{s}.npz") for s in ("fit", "es")}
    keep = {s: (np.isin(p["pdg"], (11, 13)) | (p["signal"] == 1)) if pairsel == "lepsig" else np.ones(len(p["pdg"]), bool)
            for s, p in pairs.items()}
    Pr, Pw = (torch.tensor(pairs["fit"][k][keep["fit"]][:, :nfeat].copy(), device=dev) for k in ("Xr", "Xw"))
    Per, Pew = (torch.tensor(pairs["es"][k][keep["es"]][:, :nfeat].copy(), device=dev) for k in ("Xr", "Xw"))

    def pairWeights(p, k):
        w = np.ones(int(k.sum()), np.float32)
        for item in filter(None, pairup.split(",")):
            r, wr, f = item.split(":")
            m = np.ones(len(w), bool)
            if r != "*":
                m &= p["type_r"][k] == int(r)
            if wr != "*":
                m &= p["type_w"][k] == int(wr)
            w[m] *= float(f)
        return w
    Pwt, Pewt = (torch.tensor(pairWeights(pairs[s], keep[s]), device=dev) for s in ("fit", "es"))
    print("pairs fit", len(Pr), "es", len(Per), "weighted fit sum", float(Pwt.sum()))
    pbatch = 1024


def predict(model, X):
    with torch.no_grad():
        return torch.cat([model(X[i:i + (1 << 20)].to(dev)) for i in range(0, len(X), 1 << 20)])


def wloss(z, y, w):
    return (bce(z, y) * w).sum() / w.sum()


def ploss(model, Xr, Xw, w):
    return (torch.nn.functional.softplus(model(Xw) - model(Xr)) * w).sum() / w.sum()


best, bestState, wait = 1e9, None, 0
for epoch in range(300):
    net.train()
    perm = torch.randperm(len(yt), device=dev)
    for i in range(0, len(yt), batch):
        idx = perm[i:i + batch]
        opt.zero_grad(); loss = wloss(net(Xt[idx.to(ddev)].to(dev, non_blocking=True)), yt[idx], wt[idx])
        if pairw > 0:
            pidx = torch.randint(0, len(Pr), (pbatch,), device=dev)
            loss = loss + pairw * ploss(net, Pr[pidx], Pw[pidx], Pwt[pidx])
        loss.backward(); opt.step()
    net.eval()
    vl = wloss(predict(net, Xet), yet, wet).item()
    if pairw > 0 and esmode == "sum":
        with torch.no_grad():
            vl += pairw * ploss(net, Per, Pew, Pewt).item()
    sched.step(vl)
    if vl < best - 1e-5:
        best, bestState, wait = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
    else:
        wait += 1
        if wait >= patience:
            break
net.load_state_dict(bestState); net.eval()
z = np.array(predict(net, Xet).cpu().tolist(), dtype=np.float32)  # this torch build has no NumPy bridge


def fit(sel):
    lr = LogisticRegression(C=1e6).fit(z[sel].reshape(-1, 1), ye[sel].astype(int))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


a, b = np.ones(NTYPE), np.zeros(NTYPE)
if platt == "global":
    a[:], b[:] = fit(np.ones(len(z), bool))
else:
    for k in range(NTYPE):
        sel = (te == k) & (se if platt == "type_sig" else True)
        if sel.sum() >= 200 and ye[sel].min() != ye[sel].max():
            a[k], b[k] = fit(sel)
        else:
            a[k], b[k] = fit(te == k) if (te == k).sum() >= 200 else fit(np.ones(len(z), bool))
torch.save(bestState, f"{T}/arb_{tag}.pt")
netc = Net(); netc.load_state_dict({k: v.cpu() for k, v in bestState.items()}); netc.eval()
model = Calibrated(netc, a, b).eval()
onnxFile = f"{T}/arb_{tag}.onnx"
torch.onnx.export(model, torch.tensor(Xf[:1]), onnxFile, input_names=["features"], output_names=["logit"],
                  dynamic_axes={"features": {0: "rows"}, "logit": {0: "rows"}}, opset_version=17, dynamo=False)
json.dump({"hidden": hidden, "epochs": epoch + 1, "es_loss": best, "nfeat": nfeat, "sigw": sigw, "platt": platt, "pairw": pairw, "pairsel": pairsel, "esmode": esmode, "pairfile": pairfile, "pairup": pairup,
           "seed": seed, "a": a.tolist(), "b": b.tolist(), "rows_fit": len(yf), "rows_es": len(ye)},
          open(f"{T}/arb_{tag}.json", "w"), indent=1)
print("exported", onnxFile, "epochs", epoch + 1, "es loss", best, "a", np.round(a, 3), "b", np.round(b, 3))
