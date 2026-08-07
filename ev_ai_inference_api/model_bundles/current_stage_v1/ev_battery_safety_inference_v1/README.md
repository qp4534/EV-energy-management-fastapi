# EV Battery Safety Inference Bundle v1

This is the single folder to hand to the backend/Docker team. It contains the
approved current-state classifier and safety-fusion runtime only; experiments,
training data, notebooks, and legacy Full-Sensor models are intentionally not
included.

## Contents

- `app/`: Hybrid v1 inference and v2 safety-fusion Python modules
- `models/hybrid_v1/`: immutable 30-second and 120-second HGB joblib models,
  manifest, and probability calibration
- `config/`: deployable safety policy
- `verify_bundle.py`: local load and safety-contract check
- `requirements.txt`: exact minimum runtime versions used to load the models

## What the backend must do

1. Keep one `HybridSafetySupervisorV2` instance per vehicle/battery session.
2. Call `push(sample, timestamp_seconds)` once per second.
3. Reset that session when a vehicle disconnects or an invalid sensor payload
   is received.
4. Return `ml_pattern_stage`, `physical_rule_level`, `sensor_health`,
   `final_safety_alert`, probabilities, and reason codes separately.

The model uses a 30-second warm-up, routes to the 30-second model until 119
seconds of contiguous history, then routes to the 120-second multiscale model.
It does not perform a 180-second future-risk prediction.

## Safety scope

This is a research-prototype inference bundle, not an OEM BMS or certified
fire-safety controller. `charging_gun_temperature_c` is an optional equipment
observation and must never be mapped to a cell temperature.

Before packaging into Docker, run:

```powershell
python verify_bundle.py
```
