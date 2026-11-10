"""The model evaluation of compare3.py, importable."""
import numpy as np


def predict(model, x):
    """Ratio E_gen / E_raw of a model on the features x; "a&b" is the geometric mean of models a and b."""
    if "&" in model:
        parts = [m.partition(":") for m in model.split("&")]
        w = np.array([float(p[2]) if p[2] else 1. for p in parts])
        return np.exp(np.sum([wi / w.sum() * np.log(predict(p[0], x)) for wi, p in zip(w, parts)], axis=0))
    if model.endswith(".json"):
        import xgboost as xgb
        bst = xgb.Booster()
        bst.load_model(model)
        nf = bst.num_features()
        import json
        import os
        if os.path.exists(model + ".cols"):
            return np.exp(bst.predict(xgb.DMatrix(x[:, json.load(open(model + ".cols"))["cols"]])))
        return np.exp(bst.predict(xgb.DMatrix(x[:, :nf])))
    import onnxruntime as ort
    sess = ort.InferenceSession(model)
    nf = sess.get_inputs()[0].shape[1]
    return sess.run(None, {"input": x[:, :nf]})[0].ravel()
