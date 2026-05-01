"""
demo.py
-------
End-to-end demo: register, promote, compare, roll back.

Run from the project root:
    python demo.py
"""

import json
import logging
import shutil
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlregistry.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

ARTIFACTS = Path("demo_artifacts")
STORE     = Path("demo_store")


def _train(model, X_tr, y_tr, X_te, y_te, label):
    model.fit(X_tr, y_tr)
    roc    = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    recall = float(np.mean(model.predict(X_te) == y_te))
    logger.info("[%s] ROC-AUC=%.4f  recall=%.4f", label, roc, recall)
    return {"roc_auc": round(roc, 4), "recall": round(recall, 4)}


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    if STORE.exists():
        shutil.rmtree(STORE)

    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = StandardScaler()
    X_tr = preprocessor.fit_transform(X_train)
    X_te = preprocessor.transform(X_test)
    joblib.dump(preprocessor, ARTIFACTS / "preprocessor.pkl")

    # Train v1: RandomForest
    rf   = RandomForestClassifier(n_estimators=100, random_state=42)
    m_v1 = _train(rf, X_tr, y_train, X_te, y_test, "RandomForest (v1)")
    joblib.dump(rf, ARTIFACTS / "model_v1.pkl")

    # Train v2: GradientBoosting
    gb   = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                      max_depth=4, random_state=42)
    m_v2 = _train(gb, X_tr, y_train, X_te, y_test, "GradientBoosting (v2)")
    joblib.dump(gb, ARTIFACTS / "model_v2.pkl")

    reg = ModelRegistry(STORE)

    v1 = reg.register(
        model_path=ARTIFACTS / "model_v1.pkl",
        preprocessor_path=ARTIFACTS / "preprocessor.pkl",
        metrics=m_v1,
        metadata={"model_class": "RandomForestClassifier",
                  "git_commit": "abc1234", "data_window": "last_90_days"})

    v2 = reg.register(
        model_path=ARTIFACTS / "model_v2.pkl",
        preprocessor_path=ARTIFACTS / "preprocessor.pkl",
        metrics=m_v2,
        metadata={"model_class": "GradientBoostingClassifier",
                  "git_commit": "def5678", "data_window": "last_90_days"})

    reg.promote(v1)
    logger.info("Current: %s", reg.current_version())

    comparison = reg.compare(v1, v2)
    logger.info("Comparison:\n%s", json.dumps(comparison, indent=2))

    reg.stage(v2)
    reg.promote(v2)
    logger.info("Current after challenger promotion: %s", reg.current_version())

    logger.warning("--- Simulating bad deployment: rolling back ---")
    rolled_back_to = reg.rollback()
    logger.info("Rolled back to: %s", rolled_back_to)

    print("\n-- Registry State -----------------------------------------------")
    for v in reg.list_versions():
        print(f"  {v['version']:4s}  status={v['status']:12s}  "
              f"roc_auc={v['metrics'].get('roc_auc', 'N/A')}")
    print(f"\n  Live version : {reg.current_version()}")

    # Smoke test: confirm serving code loads the rolled-back model correctly
    paths      = reg.current_artifacts()
    model_live = joblib.load(paths["model"])
    prep_live  = joblib.load(paths["preprocessor"])
    sample     = prep_live.transform(X_test[:1])
    pred       = model_live.predict(sample)[0]
    logger.info("Smoke test prediction from rolled-back model: class=%d", pred)

    shutil.rmtree(ARTIFACTS)
    shutil.rmtree(STORE)
    logger.info("Demo complete.")


if __name__ == "__main__":
    main()
