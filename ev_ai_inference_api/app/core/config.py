from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    bundle_dir: Path
    session_ttl_seconds: int = 900
    max_sessions: int = 1_000

    @classmethod
    def load(cls) -> "Settings":
        default = Path(__file__).resolve().parents[2] / "model_bundles" / "current_stage_v1" / "ev_battery_safety_inference_v1"
        return cls(Path(os.getenv("MODEL_BUNDLE_DIR", default)).resolve(), int(os.getenv("SESSION_TTL_SECONDS", "900")), int(os.getenv("MAX_SESSIONS", "1000")))


def validate_bundle(settings: Settings) -> dict:
    bundle = settings.bundle_dir
    manifest_path = bundle / "models" / "hybrid_v1" / "model_manifest.json"
    calibration_path = bundle / "models" / "hybrid_v1" / "probability_calibration.json"
    policy_path = bundle / "config" / "safety_policy.v2.json"
    for path in (manifest_path, calibration_path, policy_path):
        if not path.is_file():
            raise RuntimeError(f"required bundle artifact missing: {path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for metadata in manifest["models"].values():
        model_path = manifest_path.parent / metadata["file"]
        if not model_path.is_file():
            raise RuntimeError(f"model artifact missing: {metadata['file']}")
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if digest != metadata["sha256"]:
            raise RuntimeError(f"SHA256 mismatch: {metadata['file']}")
    if str(bundle / "app") not in sys.path:
        sys.path.insert(0, str(bundle / "app"))
    # Construct once here: joblib load and runtime feature-contract checks fail fast.
    from hybrid_safety_supervisor import HybridSafetySupervisorV2
    HybridSafetySupervisorV2(bundle / "models" / "hybrid_v1", policy_path)
    return manifest
