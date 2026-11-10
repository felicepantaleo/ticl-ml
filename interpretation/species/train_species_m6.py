# M6 neutral species model with the v2 recipe (ticlv6retrain/train_v2.py neutral: 12 inputs, 64x64 SiLU MLP, 3 logits,
# cross entropy, AdamW 2e-3 weight decay 1e-4, ReduceLROnPlateau 0.5/3, batch 2048, patience 15, at most 300 epochs,
# best es loss kept; GPU) on the neutral rows of the combined M6 tables (TenTau + e/gamma; fit and es by chain id).
# Exports the ONNX of the deployed interface (input "features" rows x 12, output "logits" rows x 3).
# usage: train_species_m6.py <tables dir> <out.onnx> [seed, default 1]
import json, sys, time
import numpy as np, torch

T, out = sys.argv[1], sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
torch.manual_seed(seed); np.random.seed(seed)
fit, es = np.load(f"{T}/fit.npz"), np.load(f"{T}/es.npz")
assert not set(np.unique(fit["chain_n"])) & set(np.unique(es["chain_n"])), "chains shared by fit and es"
Xf, yf = fit["Xn"], fit["yn"].astype(np.int64)
Xe, ye = es["Xn"], es["yn"].astype(np.int64)
mu, sd = Xf.mean(0), Xf.std(0)
sd[sd == 0] = 1.


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mu", torch.tensor(mu)); self.register_buffer("sd", torch.tensor(sd))
        self.f = torch.nn.Sequential(torch.nn.Linear(Xf.shape[1], 64), torch.nn.SiLU(), torch.nn.Linear(64, 64),
                                     torch.nn.SiLU(), torch.nn.Linear(64, 3))

    def forward(self, x):
        return self.f((x - self.mu) / self.sd)


dev = "cuda"
t0 = time.time()
net = Net().to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
lossf = torch.nn.CrossEntropyLoss()
Xt, yt = torch.tensor(Xf, device=dev), torch.tensor(yf, device=dev)
Xet, yet = torch.tensor(Xe, device=dev), torch.tensor(ye, device=dev)


def predict(X):
    with torch.no_grad():
        return torch.cat([net(X[i:i + (1 << 20)]) for i in range(0, len(X), 1 << 20)])


best, state, wait = 1e9, None, 0
for epoch in range(300):
    net.train()
    perm = torch.randperm(len(yt), device=dev)
    for i in range(0, len(yt), 2048):
        idx = perm[i:i + 2048]
        opt.zero_grad(); loss = lossf(net(Xt[idx]), yt[idx]); loss.backward(); opt.step()
    net.eval()
    vl = lossf(predict(Xet), yet).item()
    sched.step(vl)
    if vl < best - 1e-5:
        best, state, wait = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
    else:
        wait += 1
        if wait >= 15:
            break
    print(f"epoch {epoch + 1} es loss {vl:.5f}", flush=True)
net.load_state_dict(state); net.eval()
cpu = Net(); cpu.load_state_dict({k: v.cpu() for k, v in state.items()}); cpu.eval()
torch.save(state, out.replace(".onnx", ".pt"))
torch.onnx.export(cpu, torch.tensor(Xf[:1]), out, input_names=["features"], output_names=["logits"],
                  dynamic_axes={"features": {0: "rows"}, "logits": {0: "rows"}}, opset_version=17, dynamo=False)
json.dump({"seed": seed, "fit_rows": int(len(yf)), "es_rows": int(len(ye)), "epochs": epoch + 1, "es_loss": best,
           "class_fractions_fit": [float(np.mean(yf == k)) for k in range(3)], "seconds": time.time() - t0},
          open(out.replace(".onnx", ".json"), "w"))
print(f"DONE rows {len(yf)} epochs {epoch + 1} es loss {best:.5f} {time.time() - t0:.0f} s")
