# Pair tables for the D2 pairwise term, from the study dumps: pairs of scored single-track hypotheses (muon, electron,
# charged hadron, recovery) on the same track where exactly one has the type of the track particle: muon for a true
# muon, electron (with a true label: footprint of the electron) for a true electron, charged hadron or recovery for a
# true charged hadron. Splits by chain id as the pilot tables: fit id % 5 in {0, 1, 2}, es id % 5 == 3.
# usage: prepare_pairs.py <dump dir> <out dir>; output <out dir>/pairs_fit.npz, pairs_es.npz (Xr, Xw, type_r, type_w,
# pdg, signal, chain)
import glob, os, sys
from multiprocessing import Pool
import numpy as np, uproot

D, out = sys.argv[1], sys.argv[2]
CH = {211, 321, 2212, 3222, 3112, 3312, 3334}
COLS = ["event", "lumi", "type", "track", "label", "signal", "trackObjPdg", "trackSameObj", "x"]


def right(typ, pdg, label):
    if typ == 0:
        return pdg == 13
    if typ == 1:
        return pdg == 11 and label == 1
    return pdg in CH  # charged hadron (3) or recovery (6)


def chain_dirs(sample):
    """Finished chain folders of a sample, from the CHAINDIRS list (lines "<sample> <folder>")."""
    rows = (l.split() for l in open(os.environ["CHAINDIRS"]) if l.strip())
    return sorted((int(os.path.basename(d)), d) for s, d in rows if s == sample)


def chain(path):
    c = int(os.path.basename(os.path.dirname(path)))
    h = uproot.open(path)["hypDumper/hyps"].arrays(COLS, library="np")
    x = np.stack(h["x"]).astype(np.float32)
    scored = np.abs(x).sum(1) > 0
    sel = np.where(scored & (h["track"] >= 0) & np.isin(h["type"], (0, 1, 3, 6)))[0]
    key = (h["lumi"][sel].astype(np.int64) * 10**7 + h["event"][sel]) * 100000 + h["track"][sel]
    order = sel[np.argsort(key, kind="stable")]
    key = np.sort(key, kind="stable")
    R, W = [], []
    start = 0
    for end in list(np.where(np.diff(key) != 0)[0] + 1) + [len(key)]:
        g = order[start:end]; start = end
        if len(g) < 2:
            continue
        pdg = abs(int(h["trackObjPdg"][g[0]]))
        ok = np.array([right(int(h["type"][i]), pdg, int(h["label"][i])) for i in g])
        for i in g[ok]:
            for j in g[~ok]:
                R.append(i); W.append(j)
    R, W = np.array(R, dtype=np.int64), np.array(W, dtype=np.int64)
    sig = ((h["signal"] == 1) & (h["trackSameObj"] == 1)).astype(np.int8)
    return dict(Xr=x[R], Xw=x[W], type_r=h["type"][R].astype(np.int8), type_w=h["type"][W].astype(np.int8),
                pdg=np.abs(h["trackObjPdg"][R]).astype(np.int32), signal=np.maximum(sig[R], sig[W]),
                chain=np.full(len(R), c, np.int32))


if __name__ == "__main__":
    files = [f"{d}/ntuple.root" for c, d in chain_dirs(os.path.basename(D.rstrip("/")))]
    split = {"fit": [f for f in files if int(f.split("/")[-2]) % 5 in (0, 1, 2)],
             "es": [f for f in files if int(f.split("/")[-2]) % 5 == 3]}
    os.makedirs(out, exist_ok=True)
    with Pool(int(os.environ.get("NPROC", 8))) as pool:
        for s, fs in split.items():
            parts = pool.map(chain, fs)
            res = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            np.savez(f"{out}/pairs_{s}.npz", **res)
            n = len(res["pdg"])
            print(s, len(fs), "chains", n, "pairs; true track species e/mu/hadron",
                  int((res["pdg"] == 11).sum()), int((res["pdg"] == 13).sum()), n - int((res["pdg"] == 11).sum()) - int((res["pdg"] == 13).sum()),
                  "signal", int(res["signal"].sum()))
