from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np

from app.schemas.twins import TwinFrame
from app.services.cell_risk_gnn import (
    CELL_COUNT,
    FEATURE_NAMES,
    CellRiskGNNAnalyzer,
    cell_grid_position,
    frame_features,
    graph_adjacency,
)


def frame(*, hotspot_temperature_decic: int | None = None) -> TwinFrame:
    temperatures = [320] * CELL_COUNT
    if hotspot_temperature_decic is not None:
        temperatures[51] = hotspot_temperature_decic
    return TwinFrame(
        vehicle_id="car-test",
        observed_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
        sequence=1,
        temperature_decic=temperatures,
        voltage_mv=[3_800] * CELL_COUNT,
        state_level=[0] * CELL_COUNT,
        connector_temperature_decic=[300, 300, 300],
        connector_state_level=[0, 0, 0],
        hotspot_cell_index=51,
        hotspot_connector_index=0,
        ml_risk_level=0,
        physics_risk_level=0,
        final_risk_level=0,
    )


def write_constant_bundle(tmp_path, *, output_bias: list[float]) -> None:
    hidden_dim = 4
    model_path = tmp_path / "model.npz"
    np.savez_compressed(
        model_path,
        feature_names=np.asarray(FEATURE_NAMES),
        self_1_weight=np.zeros((hidden_dim, len(FEATURE_NAMES)), dtype=np.float32),
        self_1_bias=np.zeros(hidden_dim, dtype=np.float32),
        neighbor_1_weight=np.zeros((hidden_dim, len(FEATURE_NAMES)), dtype=np.float32),
        self_2_weight=np.zeros((hidden_dim, hidden_dim), dtype=np.float32),
        self_2_bias=np.zeros(hidden_dim, dtype=np.float32),
        neighbor_2_weight=np.zeros((hidden_dim, hidden_dim), dtype=np.float32),
        output_weight=np.zeros((4, hidden_dim), dtype=np.float32),
        output_bias=np.asarray(output_bias, dtype=np.float32),
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": "test-cell-gnn",
                "minimum_risk_probability": 0.5,
                "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )


def test_graph_matches_six_by_sixteen_cell_layout() -> None:
    adjacency = graph_adjacency()
    assert adjacency.shape == (96, 96)
    assert cell_grid_position(0) == (0, 0)
    assert cell_grid_position(51) == (2, 11)
    assert np.allclose(adjacency.sum(axis=1), 1.0)
    assert adjacency[0, 1] > 0
    assert adjacency[0, 4] > 0
    assert adjacency[0, 8] == 0


def test_features_use_current_cell_measurements() -> None:
    values = frame_features(frame(hotspot_temperature_decic=820))
    assert values.shape == (96, len(FEATURE_NAMES))
    assert values.dtype == np.float32
    assert values[51, 0] > values[50, 0]
    assert values[51, 1] > values[50, 1]
    assert values[51, 4] > values[50, 4]


def test_features_include_qualified_thermal_hotspot() -> None:
    source = frame().model_copy(
        update={
            "hotspot_cell_index": 37,
            "image_risk_level": 2,
            "image_confidence": 0.94,
            "image_model_status": "ready",
        }
    )
    values = frame_features(source)
    hotspot_feature = FEATURE_NAMES.index("hotspot_indicator")
    image_level_feature = FEATURE_NAMES.index("image_risk_level")
    image_confidence_feature = FEATURE_NAMES.index("image_confidence")

    assert values[:, hotspot_feature].sum() == 1.0
    assert values[37, hotspot_feature] == 1.0
    assert np.allclose(values[:, image_level_feature], 2 / 3)
    assert np.allclose(values[:, image_confidence_feature], 0.94)


def test_analyzer_marks_model_unavailable_when_bundle_is_missing(tmp_path) -> None:
    analyzer = CellRiskGNNAnalyzer.from_bundle(tmp_path)
    assert not analyzer.available
    assert analyzer.enrich_one(frame()).twin_ai_status == "unavailable"
    try:
        analyzer.require_available()
    except RuntimeError as error:
        assert "bundle is missing" in str(error)
    else:
        raise AssertionError("missing production model must fail fast")


def test_analyzer_enriches_visualization_without_changing_final_risk(tmp_path) -> None:
    write_constant_bundle(tmp_path, output_bias=[0.0, 0.0, 0.0, 5.0])
    analyzer = CellRiskGNNAnalyzer.from_bundle(tmp_path)
    source = frame()
    enriched = analyzer.enrich_one(source)

    assert analyzer.available
    assert enriched.twin_ai_status == "ready"
    assert enriched.twin_ai_model_id == "test-cell-gnn"
    assert enriched.final_risk_level == source.final_risk_level == 0
    assert enriched.cell_ai_state_level == [3] * 96
    assert enriched.state_level == [3] * 96
    assert len(enriched.cell_ai_risk_score or []) == 96
    assert len(enriched.affected_cell_indices) == 96
    assert enriched.heat_spread_direction is not None
