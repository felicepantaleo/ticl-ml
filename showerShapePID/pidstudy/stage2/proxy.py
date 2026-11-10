# Stage 2 questions on the 16 PID-study probe chains (CNN-era code), final tracksters of the interpretation stage
# (truth: StageTruthProbe stage 4, EM = dominant calo-boundary object photon, electron or pi0), E > 1 GeV:
# (1) AUC of the linking CNN, of the CLUE3DHigh shape model (shapePID_v1, its raw score) and of a BDT trained on final
#     tracksters (2 folds by chain);
# (2) species proxy (3-class BDT, neutral final tracksters, as pidstudy/scripts/ana_fin.py): deployed inputs (8 PID + 4)
#     against the shape score in place of the PID (score + 4), and the 16 shape inputs (16 + 4).
# usage: proxy.py <out.json>   (cmsenv python; onnxruntime, no torch)
import glob, json, sys
import numpy as np, uproot, xgboost as xgb, onnxruntime as ort
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
import os

# Root of the pidstudy scripts and reference models (default: this repository).
PIDSTUDY = os.environ.get("PIDSTUDY", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

dirs = sorted(d.replace("/done", "") for d in glob.glob(os.path.join(PIDSTUDY, "probe/*/0*/done")))
cols, neu = [], []
for ci, d in enumerate(dirs):
    st = uproot.open(f"{d}/ntuple.root")["stageProbe/ts"].arrays(["event", "stage", "nLC", "rawE", "domSpecies"], library="np")
    m = st["stage"] == 4
    st = {k: v[m] for k, v in st.items()}
    fi = uproot.open(f"{d}/ts_flat.root")["fin"].arrays(library="np")
    o1 = np.argsort(st["event"], kind="stable"); o2 = np.argsort(fi["event"], kind="stable")
    st = {k: v[o1] for k, v in st.items()}; fi = {k: v[o2] for k, v in fi.items()}
    assert (st["event"] == fi["event"]).all() and (st["nLC"] == fi["nLC"]).all(), d
    n = len(fi["event"])
    c = {"chain": np.full(n, ci), "dom": st["domSpecies"]}
    c.update({k: np.asarray(fi[k]) for k in fi})
    cols.append(c)
    nt = uproot.open(f"{d}/ntuple.root")["hypDumper/neutrals"].arrays(["event", "idx", "x", "species", "speciesFrac", "rawE"], library="np")
    key = {(e, i): k for k, (e, i) in enumerate(zip(fi["event"], fi["idx"]))}
    pos = np.array([key[(e, i)] for e, i in zip(nt["event"], nt["idx"])])
    nc = {"chain": np.full(len(pos), ci), "x": np.asarray(nt["x"]), "species": nt["species"], "sf": nt["speciesFrac"],
          "rawE": nt["rawE"], "pos": pos + sum(len(cc["dom"]) for cc in cols[:-1])}
    neu.append(nc)
A = {k: np.concatenate([c[k] for c in cols]) for k in cols[0]}
N = {k: np.concatenate([c[k] for c in neu]) for k in neu[0]}


def shapes(get):
    E = get("rawE"); eig = np.maximum(get("eig"), 1e-6); ax = get("axis"); eta = get("eta")
    return np.column_stack([np.log1p(E), get("rawEmE") / np.maximum(E, 1e-6), np.abs(eta), get("absZ"), np.log1p(get("nLC")),
                            np.log(eig), eig[:, 0] / eig.sum(1), get("sig"), get("sigPCA"),
                            np.abs(np.arcsinh(ax[:, 2] / np.maximum(np.hypot(ax[:, 0], ax[:, 1]), 1e-6)) - eta)]).astype(np.float32)


F = shapes(lambda k: A[k])
E = A["rawE"]; sel = (A["dom"] >= 0) & (E > 1); em = (A["dom"] == 0).astype(float)
v1 = ort.InferenceSession(os.path.join(PIDSTUDY, "shape/shapePID_v1.onnx"), providers=["CPUExecutionProvider"])
s1 = np.concatenate([v1.run(None, {"input": F[i:i + 200000]})[0].reshape(-1) for i in range(0, len(F), 200000)])
bdt = np.full(len(E), np.nan)
for f in (0, 1):
    tr = sel & (A["chain"] % 2 != f); te = (A["chain"] % 2 == f)
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=64)
    clf.fit(F[tr], em[tr]); bdt[te] = clf.predict_proba(F[te])[:, 1]
p = A["pLinkCNN"]; link = p[:, 0] + p[:, 1]
rep = {"chains": len(dirs), "final tracksters E>1": int(sel.sum())}
for lo, hi in ((1, 1e9), (1, 5), (5, 20), (20, 50), (50, 1e9)):
    m = sel & (E >= lo) & (E < hi)
    rep[f"AUC E {lo}-{hi:g}"] = {k: float(roc_auc_score(em[m], v[m])) for k, v in (("linkCNN", link), ("hiShapeModel", s1), ("finBDT", bdt))}
print(json.dumps(rep, indent=1), flush=True)
y3 = np.select([N["species"] == 22, N["species"] == 111, N["species"] == 130], [0, 1, 2], -1)
ok = (N["rawE"] > 1) & (N["sf"] >= 0.5) & (y3 >= 0)
X = N["x"]; S = F[N["pos"]]; sfin = bdt[N["pos"]][:, None]; shi = s1[N["pos"]][:, None]
sets = {"deployed (8 PID + 4)": X, "fin score + 4": np.hstack([sfin, X[:, 8:]]), "hi model score + 4": np.hstack([shi, X[:, 8:]]),
        "16 shapes + 4": np.hstack([X[:, 8:], S]), "fin score + 16 shapes + 4": np.hstack([sfin, X[:, 8:], S])}
res = {"n": int(ok.sum()), "class counts": np.bincount(y3[ok], minlength=3).tolist()}
for name, Xs in sets.items():
    q = np.full((len(y3), 3), np.nan)
    for f in (0, 1):
        tr = ok & (N["chain"] % 2 != f); te = ok & (N["chain"] % 2 == f)
        clf = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                                tree_method="hist", n_jobs=64, objective="multi:softprob", num_class=3)
        clf.fit(Xs[tr], y3[tr]); q[te] = clf.predict_proba(Xs[te])
    yy, qq = y3[ok], q[ok]
    res[name] = {"mlogloss": float(log_loss(yy, qq, labels=[0, 1, 2])), "prauc_photon": float(average_precision_score(yy == 0, qq[:, 0])),
                 "prauc_hadron": float(average_precision_score(yy == 2, qq[:, 2])), "auc_EM_vs_hadron": float(roc_auc_score(yy != 2, qq[:, 0] + qq[:, 1]))}
    print(name, res[name], flush=True)
rep["species proxy"] = res
json.dump(rep, open(sys.argv[1], "w"), indent=1)
