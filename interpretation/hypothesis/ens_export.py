# One ONNX model from several trainings of train_d2.py: the mean of their calibrated weights (a z + b of each member),
# plus an offset per hypothesis type (the one-hot slots 0-6) and a global offset. The producer runs it with
# arbitrationPlattA 1 and arbitrationPlattB 0.
# usage: ens_export.py <tables dir> <out.onnx> <global offset> <type>=<offset>,... <tag>...
import json, sys
import numpy as np, torch

T, out, glob_off, offs, tags = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5:]
NTYPE = 7


class Net(torch.nn.Module):
    def __init__(self, nin, hidden, mu, sd):
        super().__init__()
        self.register_buffer("mu", torch.tensor(mu)); self.register_buffer("sd", torch.tensor(sd))
        layers, n = [], nin
        for h in hidden:
            layers += [torch.nn.Linear(n, h), torch.nn.SiLU()]; n = h
        layers.append(torch.nn.Linear(n, 1))
        self.f = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.f((x - self.mu) / self.sd).squeeze(-1)


class Ensemble(torch.nn.Module):
    def __init__(self, nets, a, b, offset):
        super().__init__()
        self.nets = torch.nn.ModuleList(nets)
        self.register_buffer("a", torch.tensor(np.array(a), dtype=torch.float32))      # (members, types)
        self.register_buffer("b", torch.tensor(np.array(b), dtype=torch.float32))
        self.register_buffer("offset", torch.tensor(offset, dtype=torch.float32))      # (types,)

    def forward(self, x):
        t = x[:, :NTYPE]
        w = torch.stack([(t @ self.a[k]) * net(x) + t @ self.b[k] for k, net in enumerate(self.nets)]).mean(0)
        return w + t @ self.offset


nets, A, B = [], [], []
for tag in tags:
    cfg = json.load(open(f"{T}/arb_{tag}.json"))
    state = torch.load(f"{T}/arb_{tag}.pt", map_location="cpu")
    nin = state["mu"].shape[0]
    net = Net(nin, cfg["hidden"], np.zeros(nin, np.float32), np.ones(nin, np.float32))
    net.load_state_dict(state); net.eval()
    nets.append(net); A.append(cfg["a"]); B.append(cfg["b"])
offset = np.full(NTYPE, glob_off, np.float32)
for item in filter(None, offs.split(",")):
    k, v = item.split("=")
    offset[int(k)] += float(v)
model = Ensemble(nets, A, B, offset).eval()
x = torch.zeros((1, nin)); x[0, 0] = 1.
torch.onnx.export(model, x, out, input_names=["features"], output_names=["logit"],
                  dynamic_axes={"features": {0: "rows"}, "logit": {0: "rows"}}, opset_version=17, dynamo=False)
print("exported", out, "members", tags, "offset", offset)
