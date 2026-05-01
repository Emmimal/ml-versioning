"""
mlregistry/model_registry.py
------------------------------
A file-system-backed model registry that versions, promotes,
and rolls back model artifacts without requiring a database or
external service.

Windows-compatible: uses active_version.json instead of symlinks.
The JSON pointer file is written atomically (write-to-temp + rename).

Design decisions:
1. One directory per version — every artifact lives under
   store/v{n}/ with a manifest.json that captures provenance.
2. Promotion writes active_version.json atomically.
   'current' and 'previous' are keys in that file.
3. Rollback swaps current and previous in active_version.json.
4. The registry is append-only for archived versions.

Usage:
    from mlregistry.model_registry import ModelRegistry

    reg = ModelRegistry("store")
    version = reg.register(
        model_path="artifacts/model.pkl",
        preprocessor_path="artifacts/preprocessor.pkl",
        metrics={"roc_auc": 0.9312, "recall": 0.7241},
        metadata={"git_commit": "d4e8f1a", "data_window": "last_90_days"},
    )
    reg.promote(version)
    reg.rollback()
"""

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.json"
POINTER_FILE  = "active_version.json"   # replaces symlinks -- works on Windows


class RegistryError(Exception):
    pass


class ModelRegistry:
    """
    File-system model registry with versioning, promotion, and rollback.

    Directory layout after two registrations and a promotion:

        store/
        ├── v1/
        │   ├── model.pkl
        │   ├── preprocessor.pkl
        │   └── manifest.json          <- status: "archived"
        ├── v2/
        │   ├── model.pkl
        │   ├── preprocessor.pkl
        │   └── manifest.json          <- status: "production"
        └── active_version.json        <- {"current": "v2", "previous": "v1"}
    """

    def __init__(self, store_dir) -> None:
        self.store = Path(store_dir)
        self.store.mkdir(parents=True, exist_ok=True)
        logger.info("Registry initialised at %s", self.store.resolve())

    # ---- Registration -------------------------------------------------------

    def register(self, model_path, preprocessor_path,
                 metrics, metadata=None) -> str:
        """
        Copy model + preprocessor into a new versioned directory,
        write a manifest, and return the version string (e.g. 'v1').
        """
        version     = self._next_version()
        version_dir = self.store / version
        version_dir.mkdir()

        model_dst = version_dir / "model.pkl"
        prep_dst  = version_dir / "preprocessor.pkl"
        shutil.copy2(model_path, model_dst)
        shutil.copy2(preprocessor_path, prep_dst)

        manifest = {
            "version":               version,
            "registered_at":         datetime.now(timezone.utc).isoformat(),
            "model_sha256":          _sha256(model_dst),
            "preprocessor_sha256":   _sha256(prep_dst),
            "metrics":               metrics,
            "metadata":              metadata or {},
            "status":                "registered",
        }
        _write_json(version_dir / MANIFEST_FILE, manifest)

        logger.info("Registered %s | ROC-AUC %.4f",
                    version, metrics.get("roc_auc", 0))
        return version

    # ---- Staging ------------------------------------------------------------

    def stage(self, version: str) -> None:
        """Mark a version as staging -- passed validation, awaiting promotion."""
        self._update_status(version, "staging")
        logger.info("%s -> staging", version)

    # ---- Promotion ----------------------------------------------------------

    def promote(self, version: str) -> None:
        """
        Promote version to production.

        Reads the current pointer, archives the old production version,
        then atomically writes a new active_version.json.
        """
        self._assert_version_exists(version)

        pointer     = self._read_pointer()
        old_current = pointer.get("current")

        if old_current and old_current != version:
            self._update_status(old_current, "archived")

        self._write_pointer({"current": version, "previous": old_current})
        self._update_status(version, "production")

        logger.info(
            "Promoted %s -> production | previous = %s",
            version, old_current or "none",
        )

    # ---- Rollback -----------------------------------------------------------

    def rollback(self) -> str:
        """
        Swap current and previous in active_version.json.
        Returns the version that is now live.
        """
        pointer  = self._read_pointer()
        current  = pointer.get("current")
        previous = pointer.get("previous")

        if not previous:
            raise RegistryError("No previous version to roll back to.")

        self._write_pointer({"current": previous, "previous": None})
        self._update_status(previous, "production")
        if current:
            self._update_status(current, "archived")

        logger.warning("ROLLBACK: %s -> %s", current, previous)
        return previous

    # ---- Introspection ------------------------------------------------------

    def current_version(self):
        return self._read_pointer().get("current")

    def current_artifacts(self) -> dict:
        """Return paths to the currently promoted model and preprocessor."""
        version = self.current_version()
        if not version:
            raise RegistryError("No version is currently promoted.")
        version_dir = self.store / version
        return {
            "model":        version_dir / "model.pkl",
            "preprocessor": version_dir / "preprocessor.pkl",
            "manifest":     version_dir / MANIFEST_FILE,
        }

    def list_versions(self) -> list:
        """Return all registered versions sorted by version number."""
        versions = []
        for d in sorted(self.store.iterdir()):
            manifest_path = d / MANIFEST_FILE
            if d.is_dir() and manifest_path.exists():
                versions.append(_read_json(manifest_path))
        return versions

    def get_manifest(self, version: str) -> dict:
        return _read_json(self._version_dir(version) / MANIFEST_FILE)

    def compare(self, version_a: str, version_b: str) -> dict:
        """
        Side-by-side metric comparison of two versions.
        Positive delta means version_b is better than version_a.
        """
        metrics_a = self.get_manifest(version_a)["metrics"]
        metrics_b = self.get_manifest(version_b)["metrics"]

        all_keys = set(metrics_a) | set(metrics_b)
        delta = {
            k: round(metrics_b.get(k, 0) - metrics_a.get(k, 0), 6)
            for k in all_keys
        }
        return {version_a: metrics_a, version_b: metrics_b, "delta (b - a)": delta}

    def purge_archived(self, keep_last_n: int = 2) -> list:
        """
        Delete archived versions beyond the most recent N.
        Never deletes 'production' or 'staging' versions.
        """
        archived  = [v for v in self.list_versions() if v["status"] == "archived"]
        to_delete = archived[:-keep_last_n] if len(archived) > keep_last_n else []
        deleted   = []
        for v in to_delete:
            shutil.rmtree(self.store / v["version"])
            deleted.append(v["version"])
            logger.info("Purged archived version %s", v["version"])
        return deleted

    # ---- Internals ----------------------------------------------------------

    def _next_version(self) -> str:
        existing = [
            d.name for d in self.store.iterdir()
            if d.is_dir() and d.name.startswith("v")
            and (d / MANIFEST_FILE).exists()
        ]
        return f"v{len(existing) + 1}"

    def _version_dir(self, version: str) -> Path:
        d = self.store / version
        if not d.exists():
            raise RegistryError(f"Version '{version}' not found in registry.")
        return d

    def _assert_version_exists(self, version: str) -> None:
        self._version_dir(version)

    def _update_status(self, version: str, status: str) -> None:
        manifest_path = self._version_dir(version) / MANIFEST_FILE
        manifest = _read_json(manifest_path)
        manifest["status"] = status
        if status == "production":
            manifest["promoted_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(manifest_path, manifest)

    def _read_pointer(self) -> dict:
        pointer_path = self.store / POINTER_FILE
        if pointer_path.exists():
            return _read_json(pointer_path)
        return {"current": None, "previous": None}

    def _write_pointer(self, data: dict) -> None:
        """
        Write active_version.json atomically using a temp file + replace.
        Works on both Windows and POSIX.
        """
        pointer_path = self.store / POINTER_FILE
        tmp_path     = self.store / (POINTER_FILE + ".tmp")
        _write_json(tmp_path, data)
        tmp_path.replace(pointer_path)


# ---- Helpers -----------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
