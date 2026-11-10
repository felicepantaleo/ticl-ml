# Training tables of the C-2 pilot from the study dumps (dumps/<leg>/<chain>/ntuple.root, hypDumper/hyps and neutrals),
# one file per split: fit id % 5 in {0, 1, 2}, es (early stopping, Platt) id % 5 == 3.
# Hypothesis rows: every hypothesis with a footprint or tracks; X is the dumped model row (32 slots, as the producer
# computes it); y is the dumper label with the v1 rules of prepare.py for muons, neutral hadrons and charged hadrons.
# Neutral rows: the neutral final tracksters with raw E > 1 GeV and a dominant species share >= 0.5; X the 12 species
# inputs, y 0 photon, 1 pi0, 2 neutral hadron.
# usage: prepare_c2.py <leg> <out dir>; DUMPROOT: the directory that holds <leg> (default phase2/dumps)
import glob, os, sys
import numpy as np, uproot

leg, out = sys.argv[1], sys.argv[2]
P = os.environ.get("DUMPROOT", "/data/fpantale/ticlv6retrain/phase2/dumps")
CHH = [211, 321, 2212]
COLS = ["event", "type", "label", "objPdg", "purity", "trackObjPdg", "trackSameObj", "rawE", "nLC", "signal", "x"]


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


def split(i):
    return "es" if i % 5 == 3 else ("fit" if i % 5 in (0, 1, 2) else None)


os.makedirs(out, exist_ok=True)
chains = chain_dirs(leg)
acc = {s: {k: [] for k in ("X", "y", "type", "signal", "chain", "event", "Xn", "yn", "fromPhoton", "chain_n")}
       for s in ("fit", "es")}
for i, cdir in chains:
    s = split(i)
    if s is None:
        continue
    f = uproot.open(f"{cdir}/ntuple.root")
    t = f["hypDumper/hyps"].arrays(COLS, library="np")
    keep = ~((t["type"] == 6) & (t["nLC"] == 0))
    t = {k: v[keep] for k, v in t.items()}
    a = acc[s]
    a["X"].append(t["x"].astype(np.float32)); a["y"].append(label_v1(t)); a["type"].append(t["type"].astype(np.int8))
    a["signal"].append(t["signal"].astype(np.int8)); a["chain"].append(np.full(len(t["type"]), i, np.int32))
    a["event"].append(t["event"].astype(np.int32))
    n = f["hypDumper/neutrals"].arrays(["species", "speciesFrac", "rawE", "fromPhoton", "x"], library="np")
    y = np.select([n["species"] == 22, n["species"] == 111, n["species"] == 130], [0, 1, 2], -1)
    k = (n["rawE"] > 1) & (n["speciesFrac"] >= 0.5) & (y >= 0)
    a["Xn"].append(n["x"][k].astype(np.float32)); a["yn"].append(y[k].astype(np.int8))
    a["fromPhoton"].append(n["fromPhoton"][k].astype(np.int8)); a["chain_n"].append(np.full(k.sum(), i, np.int32))
for s, a in acc.items():
    np.savez(f"{out}/{s}.npz", **{k: np.concatenate(v) for k, v in a.items()})
    yn = np.concatenate(a["yn"])
    print(f"{s}: chains {len(set(np.concatenate(a['chain']).tolist()))}, hypothesis rows {sum(map(len, a['y']))} "
          f"(prevalence {np.concatenate(a['y']).mean():.3f}), neutral rows {len(yn)} "
          f"(photon {np.mean(yn == 0):.3f}, pi0 {np.mean(yn == 1):.4f}, neutral hadron {np.mean(yn == 2):.3f})")
