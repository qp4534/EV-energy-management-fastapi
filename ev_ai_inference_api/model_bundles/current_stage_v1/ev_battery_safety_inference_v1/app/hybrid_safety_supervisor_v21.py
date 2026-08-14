"""Safety Fusion v2.1 with non-duplicated hard-emergency evidence.

The HGB models, feature contract and physical thresholds are unchanged.  This
revision only prevents one derived thermal signal from serving as both the
emergency candidate and its own corroboration.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from hybrid_safety_supervisor import (
    HybridSafetySupervisorV2,
    _level_index,
    _max_level,
)


class HybridSafetySupervisorV21(HybridSafetySupervisorV2):
    """Require evidence from at least two distinct emergency groups."""

    def _physical(
        self,
        sample: Mapping[str, Any],
        timestamp: float,
    ) -> tuple[str, bool, list[str]]:
        thresholds = self.policy["physical_thresholds"]
        raw_max = self._finite(sample, "raw_temp_max_c")
        raw_mean = self._finite(sample, "raw_temp_mean_c")
        max_temp = raw_max if raw_max is not None else self._finite(
            sample, "temp_max_c"
        )
        mean_temp = raw_mean if raw_mean is not None else self._finite(
            sample, "temp_mean_c"
        )
        ambient = self._finite(sample, "ambient_temp_c")
        temp_over_ambient = (
            None
            if max_temp is None or ambient is None
            else max_temp - ambient
        )
        rate = (
            self._temperature_rate(timestamp, max_temp)
            if max_temp is not None
            else None
        )

        max_temp_level = self._threshold_level(
            max_temp, thresholds["cell_temp_max_c"]
        )
        over_ambient_level = self._threshold_level(
            temp_over_ambient, thresholds["cell_temp_over_ambient_c"]
        )
        rate_level = self._threshold_level(
            rate, thresholds["temperature_rise_rate_30s_c_per_s"]
        )
        thermal = _max_level(
            max_temp_level,
            over_ambient_level,
            rate_level,
        )

        voltages = sample.get("cell_voltages_v")
        valid_voltages: list[float] = []
        if isinstance(voltages, (list, tuple)):
            valid_voltages = [
                float(value)
                for value in voltages
                if self._finite({"x": value}, "x") is not None
            ]
        if valid_voltages:
            min_voltage, max_voltage = min(valid_voltages), max(valid_voltages)
            spread = float(
                np.percentile(valid_voltages, 95)
                - np.percentile(valid_voltages, 5)
            )
        else:
            min_voltage = max_voltage = spread = None
        electrical = _max_level(
            self._threshold_level(
                min_voltage, thresholds["min_cell_voltage_v"], "low"
            ),
            self._threshold_level(
                max_voltage, thresholds["max_cell_voltage_v"], "high"
            ),
            self._threshold_level(
                spread, thresholds["cell_voltage_spread_v"], "high"
            ),
        )

        physical = _max_level(thermal, electrical)

        # Evidence groups are intentionally non-overlapping.  In v2 the same
        # over-ambient signal could set thermal=emergency and then corroborate
        # itself.  v2.1 counts max/over-ambient together as one local-heat
        # group and requires another spatial, temporal, or electrical group.
        local_heat_emergency = (
            max_temp_level == "emergency"
            or over_ambient_level == "emergency"
        )
        mean_heat_warning = (
            mean_temp is not None
            and mean_temp >= thresholds["cell_temp_max_c"]["warning"]
        )
        rapid_rise_emergency = rate_level == "emergency"
        electrical_warning = _level_index(electrical) >= _level_index(
            "warning"
        )
        evidence = {
            "local_heat": local_heat_emergency,
            "mean_heat": mean_heat_warning,
            "rapid_rise": rapid_rise_emergency,
            "electrical": electrical_warning,
        }
        evidence_count = sum(evidence.values())
        hard_emergency = evidence_count >= 2 or (
            _level_index(thermal) >= _level_index("warning")
            and electrical_warning
        )

        reasons = [f"thermal:{thermal}", f"electrical:{electrical}"]
        if rate is not None:
            reasons.append(f"dTdt30:{rate:.3f}Cps")
        if ambient is not None and temp_over_ambient is not None:
            reasons.append(f"cell_over_ambient:{temp_over_ambient:.1f}C")
        active_evidence = [name for name, active in evidence.items() if active]
        if active_evidence:
            reasons.append("physical_evidence:" + ",".join(active_evidence))
        if hard_emergency:
            reasons.append("corroborated_physical_emergency_v21")
        elif physical == "emergency":
            reasons.append("physical_emergency_candidate_not_corroborated")
        return physical, hard_emergency, reasons

