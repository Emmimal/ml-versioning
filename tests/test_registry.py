"""
tests/test_registry.py
-----------------------
Tests for ModelRegistry: registration, promotion, rollback,
comparison, purge, and error paths.

Run:
    pytest tests/ -v
"""

import shutil
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from mlregistry.model_registry import ModelRegistry, RegistryError


# ---- Fixtures ----------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    """Fresh registry store per test."""
    return tmp_path / "registry"


@pytest.fixture
def artifacts(tmp_path):
    """Minimal valid model + preprocessor artifacts."""
    X = np.random.randn(100, 4)
    y = (X[:, 0] > 0).astype(int)

    prep  = StandardScaler()
    X_sc  = prep.fit_transform(X)
    model = LogisticRegression().fit(X_sc, y)

    prep_path  = tmp_path / "preprocessor.pkl"
    model_path = tmp_path / "model.pkl"
    joblib.dump(prep,  prep_path)
    joblib.dump(model, model_path)

    return {"model_path": model_path, "preprocessor_path": prep_path}


GOOD_METRICS   = {"roc_auc": 0.91, "recall": 0.72}
BETTER_METRICS = {"roc_auc": 0.94, "recall": 0.78}
WORSE_METRICS  = {"roc_auc": 0.80, "recall": 0.60}


# ---- Registration ------------------------------------------------------------

class TestRegistration:
    def test_first_version_is_v1(self, store, artifacts):
        reg = ModelRegistry(store)
        v = reg.register(**artifacts, metrics=GOOD_METRICS)
        assert v == "v1"

    def test_second_version_is_v2(self, store, artifacts):
        reg = ModelRegistry(store)
        reg.register(**artifacts, metrics=GOOD_METRICS)
        v2 = reg.register(**artifacts, metrics=BETTER_METRICS)
        assert v2 == "v2"

    def test_manifest_written(self, store, artifacts):
        reg = ModelRegistry(store)
        v = reg.register(**artifacts, metrics=GOOD_METRICS)
        manifest = reg.get_manifest(v)
        assert manifest["version"] == v
        assert manifest["metrics"] == GOOD_METRICS
        assert manifest["status"] == "registered"

    def test_sha256_stored_in_manifest(self, store, artifacts):
        reg = ModelRegistry(store)
        v = reg.register(**artifacts, metrics=GOOD_METRICS)
        manifest = reg.get_manifest(v)
        assert len(manifest["model_sha256"]) == 64
        assert len(manifest["preprocessor_sha256"]) == 64

    def test_artifacts_copied_not_moved(self, store, artifacts):
        reg = ModelRegistry(store)
        reg.register(**artifacts, metrics=GOOD_METRICS)
        assert artifacts["model_path"].exists()
        assert artifacts["preprocessor_path"].exists()


# ---- Promotion ---------------------------------------------------------------

class TestPromotion:
    def test_promote_sets_current(self, store, artifacts):
        reg = ModelRegistry(store)
        v1 = reg.register(**artifacts, metrics=GOOD_METRICS)
        reg.promote(v1)
        assert reg.current_version() == v1

    def test_promote_updates_status(self, store, artifacts):
        reg = ModelRegistry(store)
        v1 = reg.register(**artifacts, metrics=GOOD_METRICS)
        reg.promote(v1)
        assert reg.get_manifest(v1)["status"] == "production"

    def test_second_promotion_archives_first(self, store, artifacts):
        reg = ModelRegistry(store)
        v1 = reg.register(**artifacts, metrics=GOOD_METRICS)
        v2 = reg.register(**artifacts, metrics=BETTER_METRICS)
        reg.promote(v1)
        reg.promote(v2)
        assert reg.get_manifest(v1)["status"] == "archived"
        assert reg.get_manifest(v2)["status"] == "production"
        assert reg.current_version() == v2

    def test_current_artifacts_returns_correct_paths(self, store, artifacts):
        reg = ModelRegistry(store)
        v1 = reg.register(**artifacts, metrics=GOOD_METRICS)
        reg.promote(v1)
        paths = reg.current_artifacts()
        assert paths["model"].exists()
        assert paths["preprocessor"].exists()


# ---- Rollback ----------------------------------------------------------------

class TestRollback:
    def test_rollback_restores_previous(self, store, artifacts):
        reg = ModelRegistry(store)
        v1 = reg.register(**artifacts, metrics=GOOD_METRICS)
        v2 = reg.register(**artifacts, metrics=BETTER_METRICS)
        reg.promote(v1)
        reg.promote(v2)
        rolled = reg.rollback()
        assert rolled == v1
        assert reg.current_version() == v1

    def test_rollback_archives_bad_version(self, store, artifacts):
        reg = ModelRegistry(store)
        v1 = reg.register(**artifacts, metrics=GOOD_METRICS)
        v2 = reg.register(**artifacts, metrics=WORSE_METRICS)
        reg.promote(v1)
        reg.promote(v2)
        reg.rollback()
        assert reg.get_manifest(v2)["status"] == "archived"

    def test_rollback_without_previous_raises(self, store, artifacts):
        reg = ModelRegistry(store)
        v1 = reg.register(**artifacts, metrics=GOOD_METRICS)
        reg.promote(v1)
        with pytest.raises(RegistryError, match="No previous version"):
            reg.rollback()


# ---- Comparison --------------------------------------------------------------

class TestComparison:
    def test_compare_returns_delta(self, store, artifacts):
        reg = ModelRegistry(store)
        v1 = reg.register(**artifacts, metrics=GOOD_METRICS)
        v2 = reg.register(**artifacts, metrics=BETTER_METRICS)
        result = reg.compare(v1, v2)
        assert "delta (b - a)" in result
        assert result["delta (b - a)"]["roc_auc"] > 0   # v2 is better

    def test_compare_unknown_version_raises(self, store, artifacts):
        reg = ModelRegistry(store)
        v1 = reg.register(**artifacts, metrics=GOOD_METRICS)
        with pytest.raises(RegistryError):
            reg.compare(v1, "v99")


# ---- List & Purge ------------------------------------------------------------

class TestListAndPurge:
    def test_list_returns_all_versions(self, store, artifacts):
        reg = ModelRegistry(store)
        for _ in range(3):
            reg.register(**artifacts, metrics=GOOD_METRICS)
        assert len(reg.list_versions()) == 3

    def test_purge_archived_keeps_n(self, store, artifacts):
        reg = ModelRegistry(store)
        versions = [reg.register(**artifacts, metrics=GOOD_METRICS)
                    for _ in range(4)]
        for v in versions:
            reg.promote(v)   # each promotion archives the previous one
        reg.purge_archived(keep_last_n=1)
        archived = [v for v in reg.list_versions() if v["status"] == "archived"]
        assert len(archived) <= 1
