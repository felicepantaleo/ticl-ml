# Conflict-pair tables for the D2 pairwise term, from the study dumps: for each edge of the conflict graph (hypDumper/
# pairs) where one hypothesis is right (training label 1, the label rules of prepare_c2.py, and signal) and the other
# is wrong (training label 0), both with a model row. Splits by chain id as the pilot tables: fit id % 5 in {0, 1, 2},
# es id % 5 == 3. The chain ids of the k-th dump directory get the offset 10000 k, as in combine_tables.py.
# usage: prepare_cpairs.py <out dir> <dump dir>...; output <out dir>/pairs_conf_{fit,es}.npz (Xr, Xw, type_r, type_w,
# pdg, signal, chain); with <out dir>/pairs_{fit,es}.npz present, also pairs_both_{fit,es}.npz (these pairs plus the
# same-track pairs of true electron and muon tracks and of signal tracks).
import glob, os, sys
from collections import Counter
from multiprocessing import Pool
import numpy as np, uproot

out, dumps = sys.argv[1], sys.argv[2:]
CHH = [211, 321, 2212]
COLS = ["event", "lumi", "idx", "type", "label", "objPdg", "purity", "trackObjPdg", "trackSameObj", "rawE", "signal", "x"]


def label_v1(t):
    lab = t["label"].copy()
    nh = t["type"] == 4
    lab[nh] = (np.isin(np.abs(t["objPdg"]), [130, 2112, 310, 3122]) & (t["purity"] >= 0.5))[nh]
    mu = t["type"] == 0
    lab[mu] = (np.abs(t["trackObjPdg"]) == 13)[mu]
    ch = np.isin(t["type"], [3, 6])
    foot = (t["rawE"] <= 0) | ((t["purity"] >= 0.5) & ((t["trackObjPdg"] == 0) | (t["trackSameObj"] == 1)))
    lab[ch] = (np.isin(np.abs(t["trackObjPdg"]), CHH) & foot)[ch]
    return lab.astype(np.int8)


def chain_dirs(sample):
    """Finished chain folders of a sample, from the CHAINDIRS list (lines "<sample> <folder>")."""
    rows = (l.split() for l in open(os.environ["CHAINDIRS"]) if l.strip())
    return sorted((int(os.path.basename(d)), d) for s, d in rows if s == sample)


def chain(args):
    path, chainId = args
    f = uproot.open(path)
    t = f["hypDumper/hyps"].arrays(COLS, library="np")
    p = f["hypDumper/pairs"].arrays(["event", "lumi", "i", "j"], library="np")
    x = np.stack(t["x"]).astype(np.float32)
    lab = label_v1(t)
    scored = np.abs(x).sum(1) > 0
    right = scored & (lab == 1) & (t["signal"] == 1)
    wrong = scored & (lab == 0)
    key = t["lumi"].astype(np.int64) * 10**7 + t["event"]
    order = np.lexsort((t["idx"], key))
    skey = key[order]
    uniq, start = np.unique(skey, return_index=True)
    assert (t["idx"][order] == np.arange(len(order)) - np.repeat(start, np.diff(np.append(start, len(order))))).all()
    base = dict(zip(uniq.tolist(), start.tolist()))
    pkey = p["lumi"].astype(np.int64) * 10**7 + p["event"]
    b = np.array([base[k] for k in pkey.tolist()], dtype=np.int64)
    ri, rj = order[b + p["i"]], order[b + p["j"]]
    R = np.concatenate([ri[right[ri] & wrong[rj]], rj[right[rj] & wrong[ri]]])
    W = np.concatenate([rj[right[ri] & wrong[rj]], ri[right[rj] & wrong[ri]]])
    pdg = np.where(t["trackObjPdg"][R] != 0, np.abs(t["trackObjPdg"][R]), np.abs(t["objPdg"][R])).astype(np.int32)
    return dict(Xr=x[R], Xw=x[W], type_r=t["type"][R].astype(np.int8), type_w=t["type"][W].astype(np.int8), pdg=pdg,
                signal=np.ones(len(R), np.int8), chain=np.full(len(R), chainId, np.int32))


if __name__ == "__main__":
    split = {"fit": [], "es": []}
    for k, d in enumerate(dumps):
        for f in [f"{cd}/ntuple.root" for c, cd in chain_dirs(os.path.basename(d.rstrip("/")))]:
            c = int(f.split("/")[-2])
            s = "es" if c % 5 == 3 else ("fit" if c % 5 in (0, 1, 2) else None)
            if s:
                split[s].append((f, c + 10000 * k))
    os.makedirs(out, exist_ok=True)
    with Pool(int(os.environ.get("NPROC", 8))) as pool:
        for s, items in split.items():
            parts = pool.map(chain, items)
            res = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
            np.savez(f"{out}/pairs_conf_{s}.npz", **res)
            types = Counter(zip(res["type_r"].tolist(), res["type_w"].tolist()))
            print(s, len(items), "chains", len(res["pdg"]), "conflict pairs; (right type, wrong type) most common",
                  types.most_common(12))
            same = f"{out}/pairs_{s}.npz"
            if os.path.exists(same):
                q = dict(np.load(same))
                keep = np.isin(q["pdg"], (11, 13)) | (q["signal"] == 1)
                both = {k: np.concatenate([q[k][keep], res[k]]) for k in res}
                np.savez(f"{out}/pairs_both_{s}.npz", **both)
                print(s, "same-track pairs kept", int(keep.sum()), "both", len(both["pdg"]))
