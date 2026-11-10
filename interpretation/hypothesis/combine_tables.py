# Training tables of several samples in one directory: fit.npz, es.npz, pairs_fit.npz and pairs_es.npz of each input
# directory, concatenated; the chain ids of input k get the offset 10000 k.
# usage: combine_tables.py <out dir> <tables dir>...
import os, sys
import numpy as np

out, ins = sys.argv[1], sys.argv[2:]
os.makedirs(out, exist_ok=True)
for name in ("fit", "es", "pairs_fit", "pairs_es"):
    parts = [dict(np.load(f"{d}/{name}.npz")) for d in ins]
    for k, p in enumerate(parts):
        for key in ("chain", "chain_n"):
            if key in p:
                p[key] = p[key] + 10000 * k
    keys = parts[0].keys()
    assert all(p.keys() == keys for p in parts), f"{name}: the inputs have different columns"
    res = {key: np.concatenate([p[key] for p in parts]) for key in keys}
    np.savez(f"{out}/{name}.npz", **res)
    print(name, " + ".join(str(len(p["pdg" if name.startswith("pairs") else "y"])) for p in parts), "rows")
