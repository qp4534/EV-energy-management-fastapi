"""Safety fusion v2 for the immutable Current Best BMS Hybrid v1 package.

This module deliberately separates the ML pattern stage from independent
physical evidence and sensor health.  It is a research-prototype safety layer,
not an OEM BMS or certified vehicle safety controller.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from hybrid_runtime import HybridV1Runtime


LEVELS = ("normal", "caution", "warning", "emergency")


def _level_index(level: str) -> int:
    return LEVELS.index(level)


def _max_level(*levels: str) -> str:
    return max(levels, key=_level_index)


@dataclass(frozen=True)
class SafetyResult:
    sensor_health: str
    ml_pattern_stage: Optional[str]
    ml_probabilities: Optional[dict[str, float]]
    physical_rule_level: str
    final_safety_alert: str
    charging_equipment_observation: str
    reason_codes: list[str]
    history_seconds: int


class HybridSafetySupervisorV2:
    """1 Hz supervisor; callers must provide the six already-adapted ML signals.

    Optional raw BMS signals are used only as independent physical evidence.
    In particular, a charging-gun temperature is never treated as a cell
    temperature. Missing optional signals remove only their own rule, never
    create fabricated evidence.
    """

    def __init__(
        self,
        package_root: str | Path,
        policy_path: str | Path | None = None,
    ) -> None:
        self.runtime = HybridV1Runtime(package_root)
        policy_path = Path(policy_path or Path(__file__).with_name("safety_policy.v2.json"))
        self.policy: dict[str, Any] = json.loads(policy_path.read_text(encoding="utf-8"))
        self._temperature_history: deque[tuple[float, float]] = deque(maxlen=31)
        self._hold_until: Optional[float] = None

    def reset(self) -> None:
        self.runtime.reset()
        self._temperature_history.clear()
        self._hold_until = None

    @staticmethod
    def _finite(sample: Mapping[str, Any], name: str) -> Optional[float]:
        value = sample.get(name)
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    def _sensor_health(self, sample: Mapping[str, Any]) -> tuple[str, list[str]]:
        reasons: list[str] = []
        for name in self.policy["required_ml_signals"]:
            if self._finite(sample, name) is None:
                reasons.append(f"missing_or_invalid:{name}")
        if reasons:
            return "invalid", reasons
        if not 0 <= float(sample["voltage_v"]) <= 6:
            return "invalid", ["out_of_range:voltage_v"]
        if not -40 <= float(sample["temp_mean_c"]) <= 150 or not -40 <= float(sample["temp_max_c"]) <= 150:
            return "invalid", ["out_of_range:ml_temperature"]
        if float(sample["temp_delta_c"]) < 0:
            return "invalid", ["out_of_range:temp_delta_c"]
        if not 0 <= float(sample["temp_saturation_fraction"]) <= 1 or not 0 <= float(sample["temp_saturation_all"]) <= 1:
            return "invalid", ["out_of_range:temperature_saturation"]
        return "good", reasons

    def _temperature_rate(self, timestamp: float, temp_max: float) -> Optional[float]:
        self._temperature_history.append((timestamp, temp_max))
        if len(self._temperature_history) < 2:
            return None
        newest_time, newest_temp = self._temperature_history[-1]
        for oldest_time, oldest_temp in self._temperature_history:
            elapsed = newest_time - oldest_time
            if elapsed >= 29.0:
                return (newest_temp - oldest_temp) / elapsed
        return None

    @staticmethod
    def _threshold_level(value: Optional[float], thresholds: Mapping[str, float], direction: str = "high") -> str:
        if value is None:
            return "normal"
        if direction == "high":
            if value >= thresholds["emergency_candidate"]:
                return "emergency"
            if value >= thresholds["warning"]:
                return "warning"
            if value >= thresholds["caution"]:
                return "caution"
        else:
            if value <= thresholds["emergency_candidate"]:
                return "emergency"
            if value <= thresholds["warning"]:
                return "warning"
            if value <= thresholds["caution"]:
                return "caution"
        return "normal"

    def _physical(self, sample: Mapping[str, Any], timestamp: float) -> tuple[str, bool, list[str]]:
        t = self.policy["physical_thresholds"]
        raw_max = self._finite(sample, "raw_temp_max_c")
        raw_mean = self._finite(sample, "raw_temp_mean_c")
        max_temp = raw_max if raw_max is not None else self._finite(sample, "temp_max_c")
        mean_temp = raw_mean if raw_mean is not None else self._finite(sample, "temp_mean_c")
        ambient = self._finite(sample, "ambient_temp_c")
        temp_over_ambient = None if max_temp is None or ambient is None else max_temp - ambient
        rate = self._temperature_rate(timestamp, max_temp) if max_temp is not None else None
        thermal_levels = [
            self._threshold_level(max_temp, t["cell_temp_max_c"]),
            self._threshold_level(temp_over_ambient, t["cell_temp_over_ambient_c"]),
            self._threshold_level(rate, t["temperature_rise_rate_30s_c_per_s"]),
        ]
        thermal = _max_level(*thermal_levels)
        voltages = sample.get("cell_voltages_v")
        valid_voltages: list[float] = []
        if isinstance(voltages, (list, tuple)):
            valid_voltages = [float(value) for value in voltages if self._finite({"x": value}, "x") is not None]
        if valid_voltages:
            min_voltage, max_voltage = min(valid_voltages), max(valid_voltages)
            spread = float(np.percentile(valid_voltages, 95) - np.percentile(valid_voltages, 5))
        else:
            min_voltage = max_voltage = spread = None
        electrical = _max_level(
            self._threshold_level(min_voltage, t["min_cell_voltage_v"], "low"),
            self._threshold_level(max_voltage, t["max_cell_voltage_v"], "high"),
            self._threshold_level(spread, t["cell_voltage_spread_v"], "high"),
        )
        physical = _max_level(thermal, electrical)
        corroborated_hot = (
            (mean_temp is not None and mean_temp >= t["cell_temp_max_c"]["warning"])
            or (rate is not None and rate >= t["temperature_rise_rate_30s_c_per_s"]["emergency_candidate"])
            or (temp_over_ambient is not None and temp_over_ambient >= t["cell_temp_over_ambient_c"]["emergency_candidate"])
        )
        hard_emergency = (thermal == "emergency" and corroborated_hot) or (
            _level_index(thermal) >= _level_index("warning") and _level_index(electrical) >= _level_index("warning")
        )
        reasons = [f"thermal:{thermal}", f"electrical:{electrical}"]
        if rate is not None:
            reasons.append(f"dTdt30:{rate:.3f}Cps")
        if ambient is not None and temp_over_ambient is not None:
            reasons.append(f"cell_over_ambient:{temp_over_ambient:.1f}C")
        if hard_emergency:
            reasons.append("corroborated_physical_emergency")
        return physical, hard_emergency, reasons

    @staticmethod
    def _charging_observation(sample: Mapping[str, Any]) -> str:
        gun_temp = HybridSafetySupervisorV2._finite(sample, "charging_gun_temperature_c")
        if gun_temp is None:
            return "not_available"
        return "present_not_used_as_cell_temperature"

    def _fuse(self, ml_stage: Optional[str], physical: str, hard_emergency: bool, timestamp: float) -> tuple[str, list[str]]:
        reasons: list[str] = []
        if hard_emergency:
            self._hold_until = timestamp + self.policy["temporal"]["alert_hold_seconds"]
            return "emergency", ["hard_physical_emergency"]
        ml_effective = "normal" if ml_stage is None else min(ml_stage, "warning", key=_level_index)
        alert = _max_level(ml_effective, min(physical, "warning", key=_level_index))
        if ml_stage == "emergency":
            reasons.append("ml_emergency_requires_physical_corroboration")
        if physical == "emergency":
            reasons.append("uncorroborated_physical_emergency_capped_to_warning")
        if self._hold_until is not None and timestamp < self._hold_until:
            alert = _max_level(alert, "warning")
            reasons.append("alert_hold_active")
        return alert, reasons

    def push(self, sample: Mapping[str, Any], timestamp_seconds: float) -> SafetyResult:
        timestamp = float(timestamp_seconds)
        health, health_reasons = self._sensor_health(sample)
        if health == "invalid":
            self.reset()
            return SafetyResult("invalid", None, None, "normal", "unknown", self._charging_observation(sample), health_reasons, 0)
        ml_sample = {name: float(sample[name]) for name in self.policy["required_ml_signals"]}
        ml = self.runtime.push(ml_sample, timestamp)
        physical, hard, physical_reasons = self._physical(sample, timestamp)
        final, fusion_reasons = self._fuse(ml.stage, physical, hard, timestamp)
        return SafetyResult(
            health, ml.stage, ml.probabilities, physical, final,
            self._charging_observation(sample), physical_reasons + fusion_reasons,
            ml.history_seconds,
        )
