# ml-versioning
A file-system-backed model registry for production ML — atomic promotion, tested rollback, SHA-256 artifact integrity, and metric comparison. No database. No external service. Windows and POSIX compatible.

# ml-versioning

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/dependency-scikit--learn-F7931E)](https://scikit-learn.org/)
[![pytest](https://img.shields.io/badge/tests-16%20passed-brightgreen)](tests/)

A file-system-backed model registry for production ML — atomic promotion, tested rollback, SHA-256 artifact integrity, and side-by-side metric comparison. No database. No external service. Works on Windows and POSIX.

> Most teams skip model versioning because it feels like overhead. They discover at 11pm on a Tuesday that it was load-bearing infrastructure.

📖 **Full write-up:** [Model Versioning in Production Machine Learning](https://emitechlogic.com/model-versioning-in-production-machine-learning/)

**Series:** Production ML Engineering — Article 04 of 15 (Cluster 1: Foundation & Pipeline)

---

## What It Does

```
train() → register() → stage() → promote() → [monitor] → rollback()
                                                               ↑
                                                     active_version.json
                                                       (atomic pointer)
```

One registry, five operations:

| Operation | What It Does |
|---|---|
| `register()` | Copies model + preprocessor, computes SHA-256 checksums, writes manifest |
| `stage()` | Marks version as passed validation — awaiting promotion |
| `promote()` | Atomically swaps the active pointer, archives the previous production version |
| `compare()` | Side-by-side metric delta between any two versions |
| `rollback()` | Restores the previous production version in under one second |

---

## Installation

```bash
git clone https://github.com/Emmimal/ml-versioning.git
cd ml-versioning
pip install scikit-learn joblib numpy pytest
```

No database. No cloud account. No external service required.

---

## Quick Start

```python
from mlregistry.model_registry import ModelRegistry

reg = ModelRegistry("store")

# Register two model versions
v1 = reg.register(
    model_path="artifacts/model_v1.pkl",
    preprocessor_path="artifacts/preprocessor.pkl",
    metrics={"roc_auc": 0.9939, "recall": 0.9561},
    metadata={
        "model_class": "RandomForestClassifier",
        "git_commit": "abc1234",
        "data_window": "last_90_days"
    }
)

v2 = reg.register(
    model_path="artifacts/model_v2.pkl",
    preprocessor_path="artifacts/preprocessor.pkl",
    metrics={"roc_auc": 0.9854, "recall": 0.9386},
    metadata={"model_class": "GradientBoostingClassifier", "git_commit": "def5678"}
)

# Compare before promoting
print(reg.compare(v1, v2))

# Promote v1 to production
reg.promote(v1)

# Stage and promote challenger
reg.stage(v2)
reg.promote(v2)

# Bad deployment — roll back instantly
reg.rollback()

# Verify state
for v in reg.list_versions():
    print(f"{v['version']}  status={v['status']}  roc_auc={v['metrics']['roc_auc']}")
```

**Output:**

```
2026-05-01 08:05:47  INFO  Promoted v1 -> production | previous = none
2026-05-01 08:05:47  INFO  v2 -> staging
2026-05-01 08:05:47  INFO  Promoted v2 -> production | previous = v1
2026-05-01 08:05:47  WARNING  ROLLBACK: v2 -> v1
2026-05-01 08:05:47  INFO  Rolled back to: v1

-- Registry State -----------------------------------------------
  v1    status=production    roc_auc=0.9939
  v2    status=archived      roc_auc=0.9854

  Live version : v1
```

---

## Running the Demo

```bash
python demo.py
```

The demo runs the complete registry lifecycle end-to-end:

1. Trains RandomForest (v1) and GradientBoosting (v2) on the breast cancer dataset
2. Registers both versions with checksums and metadata
3. Promotes v1 to production
4. Compares v1 vs v2 — negative delta on both metrics
5. Promotes v2 anyway (simulating a bad deployment)
6. Rolls back to v1 automatically
7. Runs a smoke test confirming the rolled-back model serves valid predictions
8. Cleans up all artifacts

---

## Project Structure

```
ml-versioning/
├── mlregistry/
│   ├── __init__.py               # Public API — ModelRegistry, RegistryError
│   └── model_registry.py         # Core registry — no symlinks, Windows-compatible
├── registry/
│   ├── __init__.py
│   └── mlflow_bridge.py          # Optional MLflow integration layer
├── tests/
│   ├── __init__.py
│   └── test_registry.py          # 16 tests across 5 test classes
├── demo.py                       # End-to-end lifecycle demo
└── pyproject.toml
```

---

## Registry Directory Layout

After two registrations and a promotion, the store looks like this:

```
store/
├── v1/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── manifest.json          ← status: "archived", roc_auc: 0.9939
├── v2/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── manifest.json          ← status: "production", roc_auc: 0.9854
└── active_version.json        ← {"current": "v2", "previous": "v1"}
```

**One directory per version.** Every artifact is self-contained. Rollback is a single atomic file rename — no network calls, no database transactions, no manual copies.

---

## Manifest Structure

Every registered version writes a `manifest.json` capturing full provenance:

```json
{
  "version": "v1",
  "registered_at": "2026-05-01T08:05:47Z",
  "model_sha256": "a3f1b2c4d5e6f7...",
  "preprocessor_sha256": "9b8c7d6e5f4a3...",
  "metrics": {"roc_auc": 0.9939, "recall": 0.9561},
  "metadata": {
    "model_class": "RandomForestClassifier",
    "git_commit": "abc1234",
    "data_window": "last_90_days"
  },
  "status": "registered"
}
```

SHA-256 checksums are computed by streaming artifacts in 8KB chunks — safe for arbitrarily large model files.

---

## Version Lifecycle

```
registered → staging → production → archived
                ↑                      |
                └──────────────────────┘
                      rollback()
```

| Status | Meaning |
|---|---|
| `registered` | Artifacts copied and checksummed. Not yet validated. |
| `staging` | Passed the evaluation gate. Awaiting promotion. |
| `production` | Currently serving traffic. Only one version at a time. |
| `archived` | Replaced by a newer version. Retained for rollback. |

---

## Metric Comparison

```python
comparison = reg.compare(v1, v2)
```

```json
{
  "v1": {"roc_auc": 0.9939, "recall": 0.9561},
  "v2": {"roc_auc": 0.9854, "recall": 0.9386},
  "delta (b - a)": {
    "roc_auc": -0.0085,
    "recall": -0.0175
  }
}
```

Negative delta means the challenger is worse. In an automated pipeline, this feeds the evaluation gate in Stage 4 of the [retraining pipeline](https://github.com/Emmimal/ml-retraining-pipeline) — promotion is blocked if regression exceeds the configured threshold (5% by default).

---

## Rollback

```python
rolled_back_to = reg.rollback()
# WARNING: ROLLBACK: v2 -> v1
# INFO: Rolled back to: v1
```

`active_version.json` always stores two keys: `current` and `previous`. Rollback promotes `previous` to `current` and archives the old `current` — written atomically using write-to-temp-then-rename. Works on Windows and POSIX.

After rollback, the serving code loads the restored artifacts immediately:

```python
paths      = reg.current_artifacts()
model_live = joblib.load(paths["model"])
prep_live  = joblib.load(paths["preprocessor"])
pred       = model_live.predict(prep_live.transform(X_test[:1]))[0]
# INFO: Smoke test prediction from rolled-back model: class=0
```

**Note:** Rollback restores the registry pointer. It does not restart a running serving process. After calling `rollback()`, trigger a health check against `/ready`. If the serving process has not reloaded, restart it.

---

## MLflow Bridge (Optional)

`registry/mlflow_bridge.py` logs a training run to MLflow and registers the winner model in the MLflow Model Registry. The file-system registry remains the source of truth for deployment; MLflow is the experiment tracking and audit layer.

```python
from registry.mlflow_bridge import log_to_mlflow, transition_model_stage

# Log run to MLflow
run_id = log_to_mlflow(
    model_path="artifacts/model.pkl",
    preprocessor_path="artifacts/preprocessor.pkl",
    run_log_path="artifacts/run_log.json",
    experiment_name="ml-versioning-demo",
    registry_name="fraud-classifier",
)

# Promote a version in MLflow
transition_model_stage(
    registry_name="fraud-classifier",
    version=3,
    stage="Production",
)
```

Requires: `pip install mlflow scikit-learn`

---

## Running the Tests

```bash
cd ml-versioning
PYTHONPATH=. python -m pytest tests/ -v
```

```
tests/test_registry.py::TestRegistration::test_first_version_is_v1          PASSED
tests/test_registry.py::TestRegistration::test_second_version_is_v2         PASSED
tests/test_registry.py::TestRegistration::test_manifest_written             PASSED
tests/test_registry.py::TestRegistration::test_sha256_stored_in_manifest    PASSED
tests/test_registry.py::TestRegistration::test_artifacts_copied_not_moved   PASSED
tests/test_registry.py::TestPromotion::test_promote_sets_current            PASSED
tests/test_registry.py::TestPromotion::test_promote_updates_status          PASSED
tests/test_registry.py::TestPromotion::test_second_promotion_archives_first PASSED
tests/test_registry.py::TestPromotion::test_current_artifacts_returns_paths PASSED
tests/test_registry.py::TestRollback::test_rollback_restores_previous       PASSED
tests/test_registry.py::TestRollback::test_rollback_archives_bad_version    PASSED
tests/test_registry.py::TestRollback::test_rollback_without_previous_raises PASSED
tests/test_registry.py::TestComparison::test_compare_returns_delta          PASSED
tests/test_registry.py::TestComparison::test_compare_unknown_version_raises PASSED
tests/test_registry.py::TestListAndPurge::test_list_returns_all_versions    PASSED
tests/test_registry.py::TestListAndPurge::test_purge_archived_keeps_n       PASSED

16 passed in 2.64s
```

**Tests worth noting:**

- `test_rollback_without_previous_raises` — confirms rollback fails explicitly when there is no previous version, not silently
- `test_artifacts_copied_not_moved` — confirms register() never deletes the caller's source artifacts
- `test_purge_archived_keeps_n` — confirms purge never touches production or staging versions

---

## Purging Old Versions

```python
# Keep 2 most recent archived versions, delete older ones
deleted = reg.purge_archived(keep_last_n=2)
```

`purge_archived()` never deletes production or staging versions. A team retraining weekly with 500MB artifacts will accumulate ~26GB per year — keeping the last four (one month) is a reasonable default.

---

## Registry vs MLflow vs DVC vs W&B

| | **This registry** | MLflow | DVC | W&B |
|---|---|---|---|---|
| Setup | None | Medium | Low | Low |
| Infrastructure | None | Tracking server | Git + remote storage | Hosted SaaS |
| Experiment tracking | No | Yes | No | Yes |
| Data versioning | No | No | Yes | Partial |
| Rollback | Built-in | Via stage transitions | Via `git + dvc pull` | Via artifacts |
| Best for | Any team starting out | Self-hosted MLOps | Data-heavy pipelines | Research + production |

**Recommended progression:** Start here → add MLflow when multiple engineers run experiments → add DVC when training data needs versioning alongside model artifacts.

---

## When to Use This

Worth it when you have:

- Multiple model versions accumulating across retraining cycles
- A rollback requirement that needs to complete in under five minutes
- Audit requirements — regulated industries where the model behind a decision must be reconstructable
- A team large enough that not everyone knows which artifact is currently deployed

Skip it when you have:

- A single model retrained rarely with no audit requirements
- A team small enough that everyone knows the current deployment state

---

## Known Limitations

- Stores one level of rollback history — `current` and `previous` only. Multi-level rollback requires targeting a specific version manually via `promote()`.
- Token estimation in the demo uses scikit-learn's breast cancer dataset — swap in your own data for domain-specific evaluation.
- No built-in access control or approval workflows — those belong in MLflow or a managed registry if required.
- Memory is file-system only — no remote sync across machines without shared storage or a tool like DVC.

---

## Series Context

This repository is Article 04 of the Production ML Engineering series:

| Article | Repository | Topic |
|---|---|---|
| 01 | — | Production ML Engineering: The Complete Guide |
| 02 | [ml-service](https://github.com/Emmimal/ml-service) | Containerized deployment with FastAPI + Docker + CI/CD |
| 03 | [ml-retraining-pipeline](https://github.com/Emmimal/ml-retraining-pipeline) | Automated retraining with drift triggers and evaluation gate |
| **04** | **ml-versioning (this repo)** | **Model registry with atomic promotion and rollback** |
| 05 | Coming soon | Preventing catastrophic forgetting in PyTorch |

---

## References

- Zaharia et al. (2018). Accelerating the machine learning lifecycle with MLflow. *IEEE Data Engineering Bulletin.*
- Sculley et al. (2015). Hidden technical debt in machine learning systems. *NeurIPS 28.*
- Breck et al. (2017). The ML test score: A rubric for ML production readiness. *IEEE BigData.*
- Iterative AI. DVC: Data Version Control. https://dvc.org

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*Production ML Engineering — Article 04 of 15*
*Read the full write-up: [Model Versioning in Production Machine Learning](https://emitechlogic.com/model-versioning-in-production-machine-learning/)*
