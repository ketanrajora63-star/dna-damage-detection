from typing import Dict, Any, Optional
import os, numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Optional torch branch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms
except Exception:
    torch = None; nn = None; Dataset = object; DataLoader = None; models = None; transforms = None

def crossval_random_forest(X, y, cfg, work_dir) -> Dict[str, Any]:
    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    oof = np.zeros(len(y), dtype=float); metrics = []; fold = 0
    for tr, te in skf.split(X, y):
        fold += 1
        rf = RandomForestClassifier(
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth,
            min_samples_leaf=cfg.rf_min_samples_leaf,
            class_weight="balanced",
            n_jobs=cfg.n_jobs,
            random_state=cfg.random_state
        )
        clf = Pipeline([("scaler", StandardScaler(with_mean=False)), ("rf", rf)])
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:,1]; pred = (prob>=0.5).astype(int)
        oof[te] = prob
        metrics.append({
            "fold": fold,
            "acc": float(accuracy_score(y[te], pred)),
            "f1": float(f1_score(y[te], pred)),
            "precision": float(precision_score(y[te], pred)),
            "recall": float(recall_score(y[te], pred)),
            "roc_auc": float(roc_auc_score(y[te], prob)),
        })
    os.makedirs(os.path.join(work_dir,"models"), exist_ok=True)
    joblib.dump(clf, os.path.join(work_dir,"models","rf.joblib"))
    return {"oof_prob": oof.tolist(), "fold_metrics": metrics}
