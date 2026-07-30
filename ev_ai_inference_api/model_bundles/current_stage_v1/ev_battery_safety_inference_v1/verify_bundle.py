"""Fail fast if the handoff bundle cannot load or violates safety contracts."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "app"))

from hybrid_safety_supervisor import HybridSafetySupervisorV2  # noqa: E402


def sample(**overrides):
    values = {
        "voltage_v": 3.70,
        "temp_mean_c": 30.0,
        "temp_max_c": 31.0,
        "temp_delta_c": 1.0,
        "temp_saturation_fraction": 0.0,
        "temp_saturation_all": 0.0,
    }
    values.update(overrides)
    return values


def main() -> None:
    supervisor = HybridSafetySupervisorV2(
        package_root=ROOT / "models" / "hybrid_v1",
        policy_path=ROOT / "config" / "safety_policy.v2.json",
    )
    for second in range(30):
        result = supervisor.push(sample(), second)
    assert result.ml_pattern_stage in {"normal", "caution", "warning", "emergency"}
    supervisor.reset()
    hot_only = supervisor.push(sample(raw_temp_max_c=85.0, raw_temp_mean_c=30.0), 1)
    assert hot_only.final_safety_alert != "emergency"
    supervisor.reset()
    corroborated = supervisor.push(sample(raw_temp_max_c=85.0, raw_temp_mean_c=65.0), 1)
    assert corroborated.final_safety_alert == "emergency"
    print("PASS: models loaded, 30-second route active, temperature-only emergency blocked, corroborated emergency allowed")


if __name__ == "__main__":
    main()
