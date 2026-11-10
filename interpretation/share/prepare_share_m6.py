# Per-chain tables of the M6 neutral share model from the M6 dumps (energy probe with the correction off): the rows,
# 31 features and adaptive target of finalmodels/scripts/share_prepare.py (the release recipe), for TenTau (m6/tt),
# QCD (m6/qcd) and high-pT QCD (m6/qcdhi). Output m6/share_tables/<sample>/<chain>.npz: X, y and the columns for the checks.
# usage: prepare_share_m6.py; M6ROOT: the dump root (default m6), SHAREOUT: the output (default <root>/share_tables)
import glob, os
import numpy as np, uproot

M = os.environ.get("M6ROOT", "/data/fpantale/ticlv6retrain/m6d")
OUT = os.environ.get("SHAREOUT", f"{M}/share_tables")
KEEP = ["tsRegE", "tsRawE", "footTruthE", "adFootE", "adPdg", "repPdg", "domSignal", "adSignal", "tsEta", "nLC", "event"]


def chain_dirs(sample):
    """Finished chain folders of a sample, from the CHAINDIRS list (lines "<sample> <folder>")."""
    rows = (l.split() for l in open(os.environ["CHAINDIRS"]) if l.strip())
    return sorted((int(os.path.basename(d)), d) for s, d in rows if s == sample)


def table(f):
    t = uproot.open(f)["energyProbe/cands"].arrays(library="np")
    m = (t["nTk"] == 0) & (t["nTs"] > 0) & (t["tsRegE"] > 0.5) & (t["footTruthE"] > 0) & (t["candE"] >= 0.999 * t["tsRegE"])
    t = {k: v[m] for k, v in t.items()}
    cols = [np.log(t["tsRegE"]), np.log(np.maximum(t["tsRawE"], 1e-3)), t["tsRawEmE"] / np.maximum(t["tsRawE"], 1e-6),
            np.abs(t["tsEta"]), np.log1p(t["nLC"]), t["pid"], np.log1p(t["nearRawE"]), np.log1p(t["sideRawE"]), t["nNear"],
            np.log(np.maximum(t["eig"], 1e-3)), t["sigPCA"], t["sig"],
            t["lcMaxFrac"], t["lcCore2"], t["lcCore5"], t["lcRms"], t["lcZ0"], t["lcZ1"] - t["lcZ0"]]
    X = np.nan_to_num(np.column_stack(cols).astype(np.float32), nan=0., posinf=0., neginf=0.)
    assert X.shape[1] == 31
    y = np.clip(t["adFootE"] / t["footTruthE"], 0, 1).astype(np.float32)
    return dict(X=X, y=y, **{k: t[k] for k in KEEP})


for s in ("tt", "qcd", "qcdhi"):
    os.makedirs(f"{OUT}/{s}", exist_ok=True)
    chains = chain_dirs(s)
    rows = 0
    for c, cdir in chains:
        z = table(f"{cdir}/ntuple.root")
        np.savez(f"{OUT}/{s}/{c:06d}.npz", **z)
        rows += len(z["y"])
    print(s, len(chains), "chains", rows, "rows", flush=True)
