from datetime import datetime, timezone

from PIL import Image

from app.schemas.twins import ThermalInferenceResult, TwinSampleRequest
from app.services.thermal_render import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    analyze_cell_heat_scores,
    analyze_module_heat_scores,
    render_thermal_frame,
    spread_cell_heat_scores,
)
from app.services.twin_fusion import fuse_twin_state


def sample() -> TwinSampleRequest:
    temperatures = [350] * 96
    return TwinSampleRequest(
        observed_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
        sequence=1,
        temperature_decic=temperatures,
        voltage_mv=[3_800] * 96,
        connector_temperature_decic=[350, 320, 300],
        ambient_temperature_c=25.0,
        pack_current_a=90.0,
    )


def test_thermal_renderer_is_deterministic_and_has_twelve_module_rois() -> None:
    payload = sample().model_copy(
        update={"temperature_decic": [350] * 51 + [850] + [350] * 44}
    )
    first = render_thermal_frame("car-uuid-001", payload)
    second = render_thermal_frame("car-uuid-001", payload)
    assert first.image_bytes == second.image_bytes
    assert len(first.cell_heat_score) == 96
    assert first.hotspot_cell_index == 51
    assert len(first.module_heat_score) == 12
    assert first.hotspot_module_index == 6
    with Image.open(__import__("io").BytesIO(first.image_bytes)) as image:
        assert image.size == (IMAGE_WIDTH, IMAGE_HEIGHT)
        assert image.mode == "RGB"
    scores, hotspot = analyze_module_heat_scores(first.image_bytes)
    assert len(scores) == 12
    assert hotspot == 6
    cell_scores, cell_hotspot = analyze_cell_heat_scores(first.image_bytes)
    assert len(cell_scores) == 96
    assert cell_hotspot == 51


def test_heat_spread_is_one_hop_orthogonal_and_respects_module_boundaries() -> None:
    source = [0.0] * 96
    source[3] = 1.0
    spread = spread_cell_heat_scores(source)

    assert spread[3] == 1.0
    assert spread[2] == 0.35
    assert spread[7] == 0.35
    assert spread[8] == 0.20
    assert spread[9] == 0.0  # diagonal across the module boundary
    assert spread[1] == 0.0  # two cells away in the same row


def test_cell_image_fusion_does_not_broadcast_one_cell_to_a_module() -> None:
    result = ThermalInferenceResult(
        model_id="test",
        risk_level=2,
        confidence=0.95,
        probabilities=[0.02, 0.08, 0.75, 0.15],
        status="ready",
    )
    scores = [0.0] * 96
    scores[51] = 1.0
    fused = fuse_twin_state(
        sample(),
        ml_level=None,
        physical_rule_level=0,
        bms_final_level=0,
        thermal_result=result,
        thermal_cell_heat_score=tuple(scores),
    )

    assert fused["state_level"][51] == 2
    assert fused["state_level"][50] == 0
    assert fused["state_level"][52] == 0
    assert len(fused["cell_heat_score"]) == 96
    assert fused["hotspot_cell_index"] == 51


def test_image_only_danger_is_capped_but_sensor_emergency_always_wins() -> None:
    result = ThermalInferenceResult(
        model_id="test",
        risk_level=3,
        confidence=0.95,
        probabilities=[0.01, 0.04, 0.20, 0.75],
        status="ready",
    )
    image_only = fuse_twin_state(
        sample(),
        ml_level=None,
        physical_rule_level=0,
        bms_final_level=0,
        thermal_result=result,
        module_heat_score=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        hotspot_module_index=6,
    )
    assert image_only["final_risk_level"] == 2
    assert max(image_only["module_state_level"]) == 2

    physical = fuse_twin_state(
        sample().model_copy(
            update={"temperature_decic": [350] * 51 + [850] + [350] * 44}
        ),
        ml_level=None,
        physical_rule_level=3,
        bms_final_level=3,
        thermal_result=result,
        module_heat_score=(0.0,) * 12,
    )
    assert physical["final_risk_level"] == 3
    assert physical["fusion_source"] == "physics"


def test_low_confidence_image_is_marked_unqualified_and_does_not_raise_risk() -> None:
    result = ThermalInferenceResult(
        model_id="test",
        risk_level=3,
        confidence=0.42,
        probabilities=[0.10, 0.20, 0.28, 0.42],
        status="ready",
    )
    fused = fuse_twin_state(
        sample(),
        ml_level=None,
        physical_rule_level=0,
        bms_final_level=0,
        thermal_result=result,
        module_heat_score=(0.0,) * 12,
    )
    assert fused["image_model_status"] == "unqualified"
    assert fused["fusion_source"] == "image-unqualified"
    assert fused["final_risk_level"] == 0
