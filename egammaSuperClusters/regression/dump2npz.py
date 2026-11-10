#!/usr/bin/env python3
"""Per generator photon in 1.6 < |eta| < 2.8, the nearest supercluster (dR < 0.1 from the generator vertex) of each
EGammaSCFeatureDumper tree of dump.root files: raw and corrected energy and the 19 regression features.
Usage: dump2npz.py <out.npz> <label1,label2,...> <dump.root> ..."""
import sys

import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)
out, labels, files = sys.argv[1], sys.argv[2].split(","), sys.argv[3:]
NF = 19


def dphi(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def read(label):
    ch = ROOT.TChain(f"dump{label}/sc")
    for f in files:
        ch.Add(f)
    rows = {}
    for t in ch:
        key = (t.run, t.lumi, t.event)
        rows[key] = dict(nhits=t.nHits, gen=(np.array(t.genE), np.array(t.genEta), np.array(t.genPhi),
                                              np.array(t.genVz)),
                         raw=np.array(t.scRaw), e=np.array(t.scE), x=np.array(t.scX), y=np.array(t.scY),
                         z=np.array(t.scZ), f=np.array(t.scFeatures).reshape(-1, NF))
    return rows


data = {lab: read(lab) for lab in labels}
keys = sorted(data[labels[0]])
cols = {k: [] for k in ("run", "lumi", "event", "egen", "etagen", "phigen", "nhits")}
for lab in labels:
    for c in ("raw", "e", "x"):
        cols[f"{lab}_{c}"] = []
for key in keys:
    r0 = data[labels[0]][key]
    ge, geta, gphi, gvz = r0["gen"]
    for j in range(len(ge)):
        if not 1.6 < abs(geta[j]) < 2.8:
            continue
        for k, v in zip(("run", "lumi", "event", "egen", "etagen", "phigen", "nhits"),
                        (key[0], key[1], key[2], ge[j], geta[j], gphi[j], r0["nhits"])):
            cols[k].append(v)
        for lab in labels:
            r = data[lab].get(key)
            best, bdr = -1, 0.1
            if r is not None and len(r["raw"]):
                z = r["z"] - gvz[j]
                eta = np.arcsinh(z / np.hypot(r["x"], r["y"]))
                dr = np.hypot(eta - geta[j], dphi(np.arctan2(r["y"], r["x"]), gphi[j]))
                i = int(np.argmin(dr))
                if dr[i] < bdr:
                    best = i
            if best < 0:
                cols[f"{lab}_raw"].append(np.nan)
                cols[f"{lab}_e"].append(np.nan)
                cols[f"{lab}_x"].append([np.nan] * NF)
            else:
                cols[f"{lab}_raw"].append(r["raw"][best])
                cols[f"{lab}_e"].append(r["e"][best])
                cols[f"{lab}_x"].append(list(r["f"][best]))
np.savez(out, **{k: np.array(v, dtype=np.float64) for k, v in cols.items()})
print(f"{len(cols['egen'])} generator photons in {len(keys)} events written to {out}")
