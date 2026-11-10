#!/usr/bin/env python3
"""Diagnosis of the e/gamma fakes with the truth probe (EGammaTruthProbe, tree egProbe/obj and egProbe/gun).

A fake is an object with no gun particle (prompt e or gamma, 1.5 < |eta| < 3) within dR 0.2, as in
claude-reconstruction tools/egamma-gun-ab.py. The truth class of a fake:
  noise    : truth deposit < 0.5 of the reco energy of its hits
  split    : signal (gun) deposit >= 0.5 of the truth deposit (a piece of a gun shower)
  puEM     : one pile-up EM truth object (photon or electron) holds >= 0.5 of the truth deposit
             (puEMfull: that object also has >= 0.5 of its HGCAL deposit in this reco object)
  puHad    : one pile-up hadron holds >= 0.5 of the truth deposit
  mixedEM  : no truth object holds 0.5 of the truth deposit, EM truth objects hold >= 0.7 of it
  mixedHad : no truth object holds 0.5 of the truth deposit, EM truth objects hold < 0.7 of it
usage: ana_diag.py <legdir root>/eg <leg> [<leg> ...] [--out file.json]
"""
import argparse
import glob
import json
import math
import sys

import numpy as np
import uproot

KINDS = {"sc": (0, 0), "pho": (1, None), "ele": (2, None), "candG": (3, 22), "candE": (3, 11), "egts": (4, None)}
PT = [5, 10, 20, 40, 1e9]
ETA = [1.5, 2.0, 2.4, 3.0]


def load(legdir, leg):
    objs, guns = [], []
    chains = []
    for d in sorted(glob.glob(f"{legdir}/{leg}/*/done")):
        d = d[:-5]
        f = uproot.open(f"{d}/ntuple.root")
        objs.append(f["egProbe/obj"].arrays(library="np"))
        guns.append(f["egProbe/gun"].arrays(library="np"))
        chains.append(d.rstrip("/").split("/")[-1])
    cat = lambda lst: {k: np.concatenate([x[k] for x in lst]) for k in lst[0]}
    return cat(objs), cat(guns), chains


def evkey(a):
    return a["run"].astype(np.int64) * 10**12 + a["lumi"].astype(np.int64) * 10**7 + a["event"].astype(np.int64)


def gunDR(o, g):
    """Minimum dR of each object to a gun particle of the same event."""
    ko, kg = evkey(o), evkey(g)
    order = np.argsort(kg, kind="stable")
    kg_s = kg[order]
    lo = np.searchsorted(kg_s, ko, "left")
    hi = np.searchsorted(kg_s, ko, "right")
    out = np.full(len(ko), 99.)
    geta, gphi = g["eta"][order], g["phi"][order]
    for i in range(len(ko)):
        if hi[i] > lo[i]:
            de = geta[lo[i]:hi[i]] - o["eta"][i]
            dp = np.mod(gphi[lo[i]:hi[i]] - o["phi"][i] + np.pi, 2 * np.pi) - np.pi
            out[i] = np.sqrt(de * de + dp * dp).min()
    return out


def classify(o):
    dep, reco = o["dep"], np.maximum(o["recoE"], 1e-9)
    cls = np.full(len(dep), "mixedHad", dtype=object)
    cls[o["emDep"] >= 0.7 * np.maximum(dep, 1e-9)] = "mixedEM"
    domFrac = np.where(dep > 0, o["domDep"] / np.maximum(dep, 1e-9), 0.)
    em = np.isin(np.abs(o["domPdg"]), (11, 22))
    had = ~np.isin(np.abs(o["domPdg"]), (11, 22, 111, 221, 13))
    full = o["domDep"] >= 0.5 * np.maximum(o["domTot"], 1e-9)
    cls[(domFrac >= 0.5) & (o["domSignal"] == 0) & had] = "puHad"
    cls[(domFrac >= 0.5) & (o["domSignal"] == 0) & em & ~full] = "puEMpart"
    cls[(domFrac >= 0.5) & (o["domSignal"] == 0) & em & full] = "puEMfull"
    cls[o["sigDep"] >= 0.5 * np.maximum(dep, 1e-9)] = "split"
    cls[dep < 0.5 * reco] = "noise"
    return cls


CLASSES = ["split", "puEMfull", "puEMpart", "puHad", "mixedEM", "mixedHad", "noise"]


def select(o, name):
    kind, sub = KINDS[name]
    m = o["kind"] == kind
    if name == "sc":
        m &= o["coll"] == 0
    if name in ("candG", "candE"):
        m &= np.abs(o["x"][:, 0]) == sub
    return m


def objPt(o, name):
    # Superclusters use the raw transverse energy, as egamma-gun-ab.py does.
    if name == "sc":
        return o["rawE"] / np.cosh(o["eta"])
    return o["pt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("legdir")
    ap.add_argument("legs", nargs="+")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = {}
    for leg in a.legs:
        o, g, chains = load(a.legdir, leg)
        nev = len(np.unique(evkey(g)))
        dr = gunDR(o, g)
        cls = classify(o)
        res[leg] = dict(chains=chains, nev=int(nev))
        print(f"\n=== leg {leg}: {len(chains)} chains, {nev} events with gun particles, {len(g['pdg'])} gun particles")
        for name in ("sc", "pho", "ele", "candG", "candE"):
            m = select(o, name)
            if m.sum() == 0:
                continue
            pt = objPt(o, name)
            thr = {"sc": 5, "pho": 10, "ele": 5, "candG": 10, "candE": 5}[name]
            fake = m & (dr > 0.2) & (pt > thr)
            print(f"\n{name}: fakes above {thr} GeV per event {fake.sum() / nev:.3f}; by truth class (per event):")
            row = {c: float(((cls == c) & fake).sum() / nev) for c in CLASSES}
            print("   " + " ".join(f"{c:>9}" for c in CLASSES))
            print("   " + " ".join(f"{row[c]:9.3f}" for c in CLASSES))
            res[leg][name] = dict(thr=thr, total=float(fake.sum() / nev), classes=row, bins={})
            # Per pT and |eta| bin.
            for lo, hi in zip(PT[:-1], PT[1:]):
                if hi <= thr:
                    continue
                mm = fake & (pt >= max(lo, thr)) & (pt < hi)
                r = {c: float(((cls == c) & mm).sum() / nev) for c in CLASSES}
                res[leg][name]["bins"][f"pt{lo}-{hi}"] = r
                print(f"   pT {lo:>3}-{hi if hi < 1e8 else 'inf':<4} " + " ".join(f"{r[c]:9.3f}" for c in CLASSES) +
                      f"   all {mm.sum() / nev:.3f}")
            for lo, hi in zip(ETA[:-1], ETA[1:]):
                mm = fake & (np.abs(o["eta"]) >= lo) & (np.abs(o["eta"]) < hi)
                r = {c: float(((cls == c) & mm).sum() / nev) for c in CLASSES}
                res[leg][name]["bins"][f"eta{lo}-{hi}"] = r
                print(f"   eta {lo}-{hi}  " + " ".join(f"{r[c]:9.3f}" for c in CLASSES) + f"   all {mm.sum() / nev:.3f}")
            # PU EM fakes: mother of the photon.
            pe = fake & np.isin(cls, ["puEMfull", "puEMpart"])
            if pe.sum():
                mo, cnt = np.unique(np.abs(o["domMotherPdg"][pe]), return_counts=True)
                top = sorted(zip(cnt, mo), reverse=True)[:5]
                print("   PU EM fakes: mother pdg " + ", ".join(f"{m_}:{c_}" for c_, m_ in top) +
                      f"; median truth E {np.median(o['domE'][pe]):.1f} GeV, median reco/true E "
                      f"{np.median(o['e'][pe] / np.maximum(o['domE'][pe], 1e-9)):.2f}")
            mx = fake & np.isin(cls, ["mixedEM", "mixedHad"])
            if mx.sum():
                print(f"   mixed fakes: median truth objects >= 10% {np.median(o['nObj10'][mx]):.0f}, median leading share "
                      f"{np.median(o['domDep'][mx] / np.maximum(o['dep'][mx], 1e-9)):.2f}, median EM share "
                      f"{np.median(o['emDep'][mx] / np.maximum(o['dep'][mx], 1e-9)):.2f}")
            sp = fake & (cls == "split")
            if sp.sum():
                print(f"   splits: median dR to the gun {np.median(dr[sp]):.2f}, median signal object E {np.median(o['domE'][sp]):.1f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
