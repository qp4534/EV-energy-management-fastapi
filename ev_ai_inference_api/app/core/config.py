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
    cell_risk_gnn_dir: Path
    session_ttl_seconds: int = 900
    max_sessions: int = 1_000
    database_url: str = "postgresql+asyncpg://ev_app:ev_app@127.0.0.1:5433/ev_ai"
    redis_url: str = "redis://127.0.0.1:6379/0"
    cors_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    )
    twin_infra_required: bool = False
    twin_consumer_group: str = "twin-persistence"
    twin_consumer_name: str = "worker-1"
    thermal_inference_url: str = ""
    thermal_inference_token: str = ""
    thermal_inference_timeout_seconds: float = 0.8
    anomaly_persistence_enabled: bool = False
    report_jobs_enabled: bool = False
    redis_required: bool = False

    @classmethod
    def load(cls) -> "Settings":
        default = Path(__file__).resolve().parents[2] / "model_bundles" / "current_stage_v1" / "ev_battery_safety_inference_v1"
        default_cell_risk_gnn = (
            Path(__file__).resolve().parents[2]
            / "model_bundles"
            / "cell_risk_gnn_v1"
        )
        origins = tuple(
            origin.strip()
            for origin in os.getenv(
                "CORS_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173",
            ).split(",")
            if origin.strip()
        )
        required = os.getenv("TWIN_INFRA_REQUIRED", "false").strip().lower()
        anomaly_persistence = os.getenv(
            "ANOMALY_PERSISTENCE_ENABLED", "false"
        ).strip().lower()
        report_jobs = os.getenv("REPORT_JOBS_ENABLED", "false").strip().lower()
        redis_required = os.getenv("TWIN_REDIS_REQUIRED", "false").strip().lower()
        if required not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
            raise ValueError("TWIN_INFRA_REQUIRED must be a boolean")
        if anomaly_persistence not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
            raise ValueError("ANOMALY_PERSISTENCE_ENABLED must be a boolean")
        if report_jobs not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
            raise ValueError("REPORT_JOBS_ENABLED must be a boolean")
        if redis_required not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
            raise ValueError("TWIN_REDIS_REQUIRED must be a boolean")
        settings = cls(
            bundle_dir=Path(os.getenv("MODEL_BUNDLE_DIR", default)).resolve(),
            cell_risk_gnn_dir=Path(
                os.getenv("CELL_RISK_GNN_DIR", default_cell_risk_gnn)
            ).resolve(),
            session_ttl_seconds=int(os.getenv("SESSION_TTL_SECONDS", "900")),
            max_sessions=int(os.getenv("MAX_SESSIONS", "1000")),
            database_url=os.getenv(
                "DATABASE_URL",
                "postgresql+asyncpg://ev_app:ev_app@127.0.0.1:5433/ev_ai",
            ),
            redis_url=os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
            cors_origins=origins,
            twin_infra_required=required in {"1", "true", "yes", "on"},
            twin_consumer_group=os.getenv(
                "TWIN_CONSUMER_GROUP", "twin-persistence"
            ).strip(),
            twin_consumer_name=os.getenv("TWIN_CONSUMER_NAME", "worker-1").strip(),
            thermal_inference_url=os.getenv("THERMAL_INFERENCE_URL", "").strip(),
            thermal_inference_token=os.getenv("THERMAL_INFERENCE_TOKEN", "").strip(),
            thermal_inference_timeout_seconds=float(
                os.getenv("THERMAL_INFERENCE_TIMEOUT_SECONDS", "0.8")
            ),
            anomaly_persistence_enabled=anomaly_persistence
            in {"1", "true", "yes", "on"},
            report_jobs_enabled=report_jobs in {"1", "true", "yes", "on"},
            redis_required=redis_required in {"1", "true", "yes", "on"},
        )
        if settings.session_ttl_seconds <= 0:
            raise ValueError("SESSION_TTL_SECONDS must be positive")
        if settings.max_sessions <= 0:
            raise ValueError("MAX_SESSIONS must be positive")
        if not settings.database_url:
            raise ValueError("DATABASE_URL must not be empty")
        if not settings.redis_url:
            raise ValueError("REDIS_URL must not be empty")
        if not settings.twin_consumer_group or not settings.twin_consumer_name:
            raise ValueError("Twin consumer group and name must not be empty")
        if settings.thermal_inference_timeout_seconds <= 0:
            raise ValueError("THERMAL_INFERENCE_TIMEOUT_SECONDS must be positive")
        return settings


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
