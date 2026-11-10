# M6 neutral share model with the release recipe (finalmodels/scripts/train_share.py: 31 features, adaptive target,
# 64x64 SiLU MLP with a sigmoid output, unweighted MSE, AdamW 2e-3 weight decay 1e-4, ReduceLROnPlateau 0.5/3, batch
# 2048, patience 10 at 1e-7, at most 200 epochs, best es loss kept; CPU) on the TenTau, QCD and high-pT QCD tables of
# prepare_share_m6.py: fit chain id % 5 in {0, 1, 2}, es id % 5 == 3. Exports the ONNX of the deployed interface
# (input "features" rows x 31, output "fraction" rows).
# usage: train_share_m6.py <out.onnx> [seed, default 1]; SHARETABLES: the tables (default m6/share_tables)
import glob, json, os, sys, time
import numpy as np, torch

T = os.environ.get("SHARETABLES", "/data/fpantale/ticlv6retrain/m6d/share_tables")
out, seed = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 16)))


def load(part):
    files = [f for s in ("tt", "qcd", "qcdhi") for f in sorted(glob.glob(f"{T}/{s}/*.npz"))
             if (int(os.path.basename(f)[:6]) % 5 in (0, 1, 2)) == (part == "fit") and int(os.path.basename(f)[:6]) % 5 != 4]
    z = [np.load(f) for f in files]
    return np.concatenate([q["X"] for q in z]), np.concatenate([q["y"] for q in z]), len(files)


Xf, yf, nf = load("fit")
Xe, ye, ne = load("es")
mu, sd = Xf.mean(0), Xf.std(0)
sd[sd == 0] = 1.


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mu", torch.tensor(mu.astype(np.float32)))
        self.register_buffer("sd", torch.tensor(sd.astype(np.float32)))
        self.f = torch.nn.Sequential(torch.nn.Linear(31, 64), torch.nn.SiLU(), torch.nn.Linear(64, 64), torch.nn.SiLU(),
                                     torch.nn.Linear(64, 1))

    def forward(self, x):
        return torch.sigmoid(self.f((x - self.mu) / self.sd)).squeeze(-1)


torch.manual_seed(seed)
t0 = time.time()
net = Net()
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
Xt, yt, Xet, yet = (torch.tensor(a) for a in (Xf, yf, Xe, ye))
lossf = lambda p, y: torch.mean((p - y) ** 2)
best, state, wait = 1e9, None, 0
for epoch in range(200):
    net.train()
    perm = torch.randperm(len(yt))
    for i in range(0, len(yt), 2048):
        idx = perm[i:i + 2048]
        opt.zero_grad(); loss = lossf(net(Xt[idx]), yt[idx]); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        vl = lossf(net(Xet), yet).item()
    sched.step(vl)
    if vl < best - 1e-7:
        best, state, wait = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
    else:
        wait += 1
        if wait >= 10:
            break
    print(f"epoch {epoch + 1} es MSE {vl:.6f}", flush=True)
net.load_state_dict(state); net.eval()
torch.save(state, out.replace(".onnx", ".pt"))
torch.onnx.export(net, torch.tensor(Xf[:1]), out, input_names=["features"], output_names=["fraction"],
                  dynamic_axes={"features": {0: "rows"}, "fraction": {0: "rows"}}, opset_version=17, dynamo=False)
json.dump({"seed": seed, "fit_chains": nf, "es_chains": ne, "fit_rows": int(len(yf)), "es_rows": int(len(ye)),
           "epochs": epoch + 1, "es_mse": best, "seconds": time.time() - t0}, open(out.replace(".onnx", ".json"), "w"))
print(f"DONE fit chains {nf} rows {len(yf)} es chains {ne} epochs {epoch + 1} es MSE {best:.6f} {time.time() - t0:.0f} s")
