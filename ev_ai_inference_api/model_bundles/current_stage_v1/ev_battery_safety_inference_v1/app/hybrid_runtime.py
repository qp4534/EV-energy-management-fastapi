"""Standalone 1 Hz inference runtime for Current Best BMS Hybrid v1.

This module deliberately performs *only* the ML routing and probability
calibration contract.  It does not implement FastAPI, the 180-second onset
model, OOD detection, or final safety-alert fusion.
"""
from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

# Avoid joblib's Windows physical-core probe during lightweight inference.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import numpy as np
import pandas as pd


CLASS_NAMES = ("normal", "caution", "warning", "emergency")
SIGNALS = (
    "voltage_v",
    "temp_mean_c",
    "temp_max_c",
    "temp_delta_c",
    "temp_saturation_fraction",
    "temp_saturation_all",
)
WINDOWS_BY_ROUTE = {"stage_30s": (5, 15, 30), "stage_120s": (5, 15, 30, 60, 120)}


@dataclass(frozen=True)
class RuntimePrediction:
    """ML-only current-state result; it is not the final safety alert."""

    status: str
    history_seconds: int
    model_route: Optional[str]
    stage: Optional[str]
    probabilities: Optional[dict[str, float]]
    calibrated: bool


def default_package_root() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "deployment_bms_v1"
        / "current_best_bms_hybrid_v1"
    )


def _expected_feature_names(windows: tuple[int, ...]) -> list[str]:
    names: list[str] = []
    for signal in SIGNALS:
        names.extend((f"{signal}__current", f"{signal}__diff_1s"))
        for window in windows:
            names.extend(
                (
                    f"{signal}__mean_{window}s",
                    f"{signal}__std_{window}s",
                    f"{signal}__min_{window}s",
                    f"{signal}__max_{window}s",
                    f"{signal}__delta_{window}s",
                    f"{signal}__endpoint_slope_{window}s",
                )
            )
    return names


def _temperature_scale(raw: np.ndarray, temperature: float) -> np.ndarray:
    """Exact package contract: softmax(log(clip(p))/T)."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    logits = np.log(np.clip(raw.astype(float), 1e-12, 1.0)) / temperature
    logits -= logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


class HybridV1Runtime:
    """Keeps a contiguous 1 Hz history and routes it to the packaged model.

    Each ``push`` input must contain the six already-adapted causal signals
    listed in ``SIGNALS``.  Temperature saturation values are adapter outputs
    in [0, 1], not raw temperatures.  A timestamp gap other than one second
    resets history rather than silently fabricating missing observations.
    """

    def __init__(self, package_root: str | Path | None = None) -> None:
        self.package_root = Path(package_root or default_package_root()).resolve()
        manifest_path = self.package_root / "model_manifest.json"
        calibration_path = self.package_root / "probability_calibration.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        self.models = {
            route: joblib.load(self.package_root / metadata["file"])
            for route, metadata in self.manifest["models"].items()
        }
        self.temperatures = {
            "stage_30s": float(calibration["temperatures"]["stage_30s"]),
            "stage_120s": float(calibration["temperatures"]["stage_120s"]),
        }
        self._verify_bundles()
        self._history: deque[dict[str, float]] = deque(maxlen=120)
        self._last_timestamp: Optional[float] = None

    def reset(self) -> None:
        self._history.clear()
        self._last_timestamp = None

    def _verify_bundles(self) -> None:
        for route, bundle in self.models.items():
            if tuple(bundle["class_names"]) != CLASS_NAMES:
                raise RuntimeError(f"{route}: unexpected class order")
            expected = _expected_feature_names(WINDOWS_BY_ROUTE[route])
            # Training selected columns alphabetically before fitting.  Check
            # the exact generated name set here; `_features` then deliberately
            # emits the bundle's saved order for sklearn prediction.
            if len(bundle["feature_columns"]) != len(expected) or set(
                bundle["feature_columns"]
            ) != set(expected):
                raise RuntimeError(
                    f"{route}: model feature contract differs from runtime"
                )
            if not hasattr(bundle["model"], "predict_proba"):
                raise RuntimeError(f"{route}: model has no predict_proba")

    @staticmethod
    def _validate(sample: Mapping[str, float]) -> dict[str, float]:
        missing = [name for name in SIGNALS if name not in sample]
        extra = [name for name in sample if name not in SIGNALS]
        if missing or extra:
            raise ValueError(f"six-signal contract violation; missing={missing}, extra={extra}")
        values = {name: float(sample[name]) for name in SIGNALS}
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("all signals must be finite")
        if not 0.0 <= values["voltage_v"] <= 6.0:
            raise ValueError("voltage_v must be in [0, 6]")
        for name in ("temp_mean_c", "temp_max_c"):
            if not -40.0 <= values[name] <= 150.0:
                raise ValueError(f"{name} must be in [-40, 150]")
        if values["temp_delta_c"] < 0.0:
            raise ValueError("temp_delta_c must be non-negative")
        for name in ("temp_saturation_fraction", "temp_saturation_all"):
            if not 0.0 <= values[name] <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        return values

    def _features(self, route: str) -> pd.DataFrame:
        windows = WINDOWS_BY_ROUTE[route]
        frame = pd.DataFrame(list(self._history), columns=SIGNALS)
        row: dict[str, float] = {}
        for signal in SIGNALS:
            values = frame[signal].to_numpy(dtype=float)
            row[f"{signal}__current"] = values[-1]
            row[f"{signal}__diff_1s"] = values[-1] - values[-2]
            for window in windows:
                tail = values[-window:]
                delta = tail[-1] - tail[0]
                row[f"{signal}__mean_{window}s"] = float(tail.mean())
                row[f"{signal}__std_{window}s"] = float(tail.std(ddof=0))
                row[f"{signal}__min_{window}s"] = float(tail.min())
                row[f"{signal}__max_{window}s"] = float(tail.max())
                row[f"{signal}__delta_{window}s"] = float(delta)
                row[f"{signal}__endpoint_slope_{window}s"] = float(delta / (window - 1))
        columns = self.models[route]["feature_columns"]
        return pd.DataFrame([[row[column] for column in columns]], columns=columns)

    def push(
        self, sample: Mapping[str, float], timestamp_seconds: Optional[float] = None
    ) -> RuntimePrediction:
        values = self._validate(sample)
        if timestamp_seconds is not None:
            timestamp_seconds = float(timestamp_seconds)
            if not np.isfinite(timestamp_seconds):
                raise ValueError("timestamp_seconds must be finite")
            if self._last_timestamp is not None and not np.isclose(
                timestamp_seconds - self._last_timestamp, 1.0, atol=1e-6
            ):
                self.reset()
        self._history.append(values)
        self._last_timestamp = timestamp_seconds
        count = len(self._history)
        if count < 30:
            return RuntimePrediction("warming_up", count, None, None, None, False)
        route = "stage_120s" if count >= 120 else "stage_30s"
        features = self._features(route)
        raw = self.models[route]["model"].predict_proba(features)[0]
        calibrated = _temperature_scale(raw, self.temperatures[route])
        stage_index = int(np.argmax(calibrated))
        return RuntimePrediction(
            "ok", count, route, CLASS_NAMES[stage_index],
            {name: float(value) for name, value in zip(CLASS_NAMES, calibrated)}, True,
        )
