from __future__ import annotations

from app.schemas.twins import ThermalInferenceResult, TwinSampleRequest

from .thermal_render import module_scores_from_cells, sensor_cell_heat_scores


def temperature_level(temperature_decic: int) -> int:
    if temperature_decic >= 800:
        return 3
    if temperature_decic >= 600:
        return 2
    if temperature_decic >= 450:
        return 1
    return 0


def voltage_level(voltage_mv: int) -> int:
    if voltage_mv <= 1_500 or voltage_mv >= 4_500:
        return 3
    if voltage_mv <= 2_000 or voltage_mv >= 4_400:
        return 2
    if voltage_mv < 2_400 or voltage_mv > 4_350:
        return 1
    return 0


def _image_module_level(score: float, image_level: int) -> int:
    if score < 0.25:
        local_level = 0
    elif score < 0.50:
        local_level = 1
    elif score < 0.75:
        local_level = 2
    else:
        local_level = 3
    return min(local_level, image_level)


def fuse_twin_state(
    payload: TwinSampleRequest,
    *,
    ml_level: int | None,
    physical_rule_level: int,
    bms_final_level: int,
    thermal_result: ThermalInferenceResult | None = None,
    thermal_cell_heat_score: tuple[float, ...] | None = None,
    module_heat_score: tuple[float, ...] | None = None,
    hotspot_module_index: int | None = None,
    thermal_frame_ref: str | None = None,
    thermal_frame_sha256: str | None = None,
    minimum_image_confidence: float = 0.70,
) -> dict[str, object]:
    if len(payload.temperature_decic) != 96 or len(payload.voltage_mv) != 96:
        raise ValueError("twin sample must contain 96 temperature and voltage values")
    sensor_cell_levels = [
        max(temperature_level(temperature), voltage_level(voltage))
        for temperature, voltage in zip(
            payload.temperature_decic,
            payload.voltage_mv,
            strict=True,
        )
    ]
    connector_levels = [
        temperature_level(value) for value in payload.connector_temperature_decic
    ]
    sensor_module_levels = [max(sensor_cell_levels[index : index + 8]) for index in range(0, 96, 8)]
    ambient = (
        float(payload.ambient_temperature_c)
        if payload.ambient_temperature_c is not None
        else 25.0
    )
    sensor_heat_score = sensor_cell_heat_scores(payload.temperature_decic, ambient)
    physics_level = max(
        physical_rule_level,
        max(sensor_cell_levels),
        max(connector_levels),
    )
    sensor_final_level = max(bms_final_level, physics_level)

    result = thermal_result or ThermalInferenceResult(status="unavailable")
    image_is_valid = (
        result.status == "ready"
        and result.risk_level is not None
        and result.confidence is not None
        and result.confidence >= minimum_image_confidence
    )
    image_status = result.status
    if (
        result.status == "ready"
        and result.confidence is not None
        and result.confidence < minimum_image_confidence
    ):
        image_status = "unqualified"
    image_level = result.risk_level if image_is_valid and result.risk_level is not None else 0
    effective_image_level = (
        min(image_level, 2) if image_is_valid and sensor_final_level == 0 else image_level
    )
    if thermal_cell_heat_score is not None and len(thermal_cell_heat_score) != 96:
        raise ValueError("thermal cell heat score array must contain 96 values")
    if module_heat_score is not None and len(module_heat_score) != 12:
        raise ValueError("module heat score array must contain 12 values")
    image_cell_levels = [0] * 96
    valid_cell_scores = thermal_cell_heat_score is not None
    if image_is_valid and valid_cell_scores:
        image_cell_levels = [
            _image_module_level(score, effective_image_level)
            for score in thermal_cell_heat_score or ()
        ]
    fused_cell_levels = [
        max(sensor_level, image_level)
        for sensor_level, image_level in zip(
            sensor_cell_levels, image_cell_levels, strict=True
        )
    ]
    module_levels = [
        max(fused_cell_levels[index : index + 8]) for index in range(0, 96, 8)
    ]
    # Keep accepting the pre-cell ROI input for older callers, but only use it
    # as a 12-module summary. It is never copied back into the 96 cell levels.
    if image_is_valid and not valid_cell_scores and module_heat_score is not None:
        module_levels = [
            max(sensor_level, _image_module_level(score, effective_image_level))
            for sensor_level, score in zip(sensor_module_levels, module_heat_score, strict=True)
        ]
    final_level = max(sensor_final_level, effective_image_level)
    if physics_level >= 3:
        final_level = 3

    if physics_level >= 3:
        fusion_source = "physics"
    elif image_status == "unqualified":
        fusion_source = "image-unqualified"
    elif image_is_valid:
        fusion_source = "image+sensor"
    else:
        fusion_source = "sensor-only"

    image_heat_scores = [0.0] * 96
    if image_is_valid and valid_cell_scores:
        image_limit = effective_image_level / 3.0
        image_heat_scores = [
            min(image_limit, max(0.0, min(1.0, float(score))))
            for score in thermal_cell_heat_score or ()
        ]
    cell_heat_scores = [
        max(sensor_score, image_score, state_level / 3.0)
        for sensor_score, image_score, state_level in zip(
            sensor_heat_score, image_heat_scores, fused_cell_levels, strict=True
        )
    ]
    summary_module_heat = module_scores_from_cells(cell_heat_scores)
    if not valid_cell_scores and module_heat_score is not None:
        summary_module_heat = tuple(float(score) for score in module_heat_score)
    hotspot_cell = max(range(96), key=cell_heat_scores.__getitem__)
    module_hotspot = hotspot_cell // 8
    if not valid_cell_scores and hotspot_module_index is not None:
        module_hotspot = hotspot_module_index

    return {
        "state_level": fused_cell_levels,
        "connector_state_level": connector_levels,
        "physics_risk_level": physics_level,
        "final_risk_level": final_level,
        "image_risk_level": result.risk_level,
        "image_confidence": result.confidence,
        "image_probabilities": result.probabilities,
        "image_model_status": image_status,
        "cell_heat_score": cell_heat_scores,
        "module_heat_score": list(summary_module_heat),
        "module_state_level": module_levels,
        "hotspot_cell_index": hotspot_cell,
        "hotspot_module_index": module_hotspot,
        "thermal_frame_ref": thermal_frame_ref,
        "thermal_frame_sha256": thermal_frame_sha256,
        "fusion_source": fusion_source,
    }
