import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def sample(timestamp: int, **overrides):
    value = {"timestamp_seconds": timestamp, "voltage_v": 3.92, "temp_mean_c": 39.4, "temp_max_c": 43.1, "temp_delta_c": 3.7, "temp_saturation_fraction": 0.0, "temp_saturation_all": False}
    value.update(overrides)
    return value
