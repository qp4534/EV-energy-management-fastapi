from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from app.schemas.twins import CELL_COUNT, TwinFrame


MODEL_ID = "cell-risk-graphsage-v1"
FEATURE_NAMES = (
    "temperature_absolute",
    "temperature_delta",
    "voltage_absolute",
    "voltage_delta",
    "thermal_heat",
    "hotspot_indicator",
    "bms_risk_level",
    "image_risk_level",
    "image_confidence",
)
CELLS_PER_MODULE = 8
MODULE_COLUMNS = 4
MODULE_CELL_ROWS = 2
MODULE_CELL_COLUMNS = 4


def cell_grid_position(index: int) -> tuple[int, int]:
    """Map the concept pack's cell order to its physical 6x16 grid."""

    if not 0 <= index < CELL_COUNT:
        raise ValueError("cell index must be between 0 and 95")
    module_row, module_column = divmod(index // CELLS_PER_MODULE, MODULE_COLUMNS)
    cell_row, cell_column = divmod(index % CELLS_PER_MODULE, MODULE_CELL_COLUMNS)
    return (
        module_row * MODULE_CELL_ROWS + cell_row,
        module_column * MODULE_CELL_COLUMNS + cell_column,
    )


def graph_adjacency() -> np.ndarray:
    """Return row-normalized orthogonal-neighbor adjacency without self loops."""

    positions = [cell_grid_position(index) for index in range(CELL_COUNT)]
    by_position = {position: index for index, position in enumerate(positions)}
    adjacency = np.zeros((CELL_COUNT, CELL_COUNT), dtype=np.float32)
    for index, (row, column) in enumerate(positions):
        for row_delta, column_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = by_position.get((row + row_delta, column + column_delta))
            if neighbor is not None:
                adjacency[index, neighbor] = 1.0
    degrees = np.maximum(adjacency.sum(axis=1, keepdims=True), 1.0)
    return adjacency / degrees


@dataclass(frozen=True)
class CellRiskGraphSAGEWeights:
    """NumPy weights for the production GraphSAGE inference runtime."""

    self_1_weight: np.ndarray
    self_1_bias: np.ndarray
    neighbor_1_weight: np.ndarray
    self_2_weight: np.ndarray
    self_2_bias: np.ndarray
    neighbor_2_weight: np.ndarray
    output_weight: np.ndarray
    output_bias: np.ndarray

    def logits(self, features: np.ndarray) -> np.ndarray:
        adjacency = graph_adjacency()
        neighbors = np.einsum("ij,bjf->bif", adjacency, features, optimize=True)
        hidden = np.maximum(
            features @ self.self_1_weight.T
            + self.self_1_bias
            + neighbors @ self.neighbor_1_weight.T,
            0.0,
        )
        hidden_neighbors = np.einsum(
            "ij,bjf->bif", adjacency, hidden, optimize=True
        )
        hidden = np.maximum(
            hidden @ self.self_2_weight.T
            + self.self_2_bias
            + hidden_neighbors @ self.neighbor_2_weight.T,
            0.0,
        )
        return hidden @ self.output_weight.T + self.output_bias


def frame_features(frame: TwinFrame) -> np.ndarray:
    """Build the per-cell model features without using rule-derived cell labels."""

    temperatures = np.asarray(frame.temperature_decic, dtype=np.float32) / 10.0
    voltages = np.asarray(frame.voltage_mv, dtype=np.float32) / 1_000.0
    temperature_mean = float(temperatures.mean())
    voltage_median = float(np.median(voltages))
    ambient = min(temperature_mean, 30.0)
    # Do not reuse TwinFrame.cell_heat_score here. That field already contains
    # rule-derived cell state and would leak the training target into the model.
    thermal_heat = np.clip(
        (temperatures - ambient - 5.0) / 75.0,
        0.0,
        1.0,
    )
    measurement_salience = np.maximum(
        np.clip((temperatures - float(np.median(temperatures))) / 40.0, 0.0, 1.0),
        np.clip(
            np.abs(voltages - float(np.median(voltages))) / 1.5,
            0.0,
            1.0,
        ),
    )
    hotspot_index = (
        frame.hotspot_cell_index
        if frame.image_model_status == "ready"
        and frame.image_risk_level is not None
        else int(np.argmax(measurement_salience))
    )
    hotspot_indicator = np.zeros(CELL_COUNT, dtype=np.float32)
    hotspot_indicator[hotspot_index] = 1.0
    image_level = float(frame.image_risk_level or 0) / 3.0
    image_confidence = (
        float(frame.image_confidence or 0.0)
        if frame.image_model_status == "ready"
        else 0.0
    )
    features = np.stack(
        (
            np.clip((temperatures - 40.0) / 30.0, -2.0, 4.0),
            np.clip((temperatures - temperature_mean) / 20.0, -3.0, 5.0),
            np.clip((voltages - 3.7) / 0.7, -5.0, 4.0),
            np.clip((voltages - voltage_median) / 0.5, -6.0, 6.0),
            np.clip(thermal_heat, 0.0, 1.0),
            hotspot_indicator,
            np.full(CELL_COUNT, frame.final_risk_level / 3.0, dtype=np.float32),
            np.full(CELL_COUNT, image_level, dtype=np.float32),
            np.full(CELL_COUNT, image_confidence, dtype=np.float32),
        ),
        axis=-1,
    )
    return features.astype(np.float32, copy=False)


@dataclass(frozen=True)
class CellRiskAnalysis:
    model_id: str
    risk_score: list[float]
    state_level: list[int]
    hotspot_cell_index: int
    affected_cell_indices: list[int]
    heat_spread_direction: str


def _direction(
    hotspot: int,
    affected: list[int],
    scores: np.ndarray,
) -> str:
    if not affected:
        return "stable"
    neighbors = [index for index in affected if index != hotspot]
    if not neighbors:
        return "localized"
    hotspot_row, hotspot_column = cell_grid_position(hotspot)
    weights = np.asarray([max(float(scores[index]), 1e-6) for index in neighbors])
    row_delta = float(
        np.average(
            [cell_grid_position(index)[0] - hotspot_row for index in neighbors],
            weights=weights,
        )
    )
    column_delta = float(
        np.average(
            [cell_grid_position(index)[1] - hotspot_column for index in neighbors],
            weights=weights,
        )
    )
    if float(np.hypot(row_delta, column_delta)) < 0.55:
        return "outward"
    vertical = "front" if row_delta < -0.35 else "rear" if row_delta > 0.35 else ""
    horizontal = "left" if column_delta < -0.35 else "right" if column_delta > 0.35 else ""
    if vertical and horizontal:
        return f"{vertical}-{horizontal}"
    return vertical or horizontal or "localized"


class CellRiskGNNAnalyzer:
    """Loads the trained cell graph model and enriches TwinFrames for 3D use."""

    def __init__(
        self,
        model: CellRiskGraphSAGEWeights | None = None,
        *,
        model_id: str = MODEL_ID,
        minimum_risk_probability: float = 0.50,
    ) -> None:
        self.model = model
        self.model_id = model_id
        self.minimum_risk_probability = minimum_risk_probability

    @property
    def available(self) -> bool:
        return self.model is not None

    def require_available(self) -> None:
        if not self.available:
            raise RuntimeError(
                "cell risk GNN bundle is missing; model.npz and manifest.json are required"
            )

    @classmethod
    def from_bundle(cls, bundle_dir: Path | str) -> "CellRiskGNNAnalyzer":
        bundle = Path(bundle_dir)
        manifest_path = bundle / "manifest.json"
        model_path = bundle / "model.npz"
        if not manifest_path.is_file() or not model_path.is_file():
            return cls()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if digest != manifest.get("sha256"):
            raise RuntimeError("cell risk GNN SHA256 does not match its manifest")
        checkpoint = np.load(model_path, allow_pickle=False)
        feature_names = tuple(str(value) for value in checkpoint["feature_names"])
        if feature_names != FEATURE_NAMES:
            raise RuntimeError("cell risk GNN feature contract does not match runtime")
        model = CellRiskGraphSAGEWeights(
            self_1_weight=checkpoint["self_1_weight"],
            self_1_bias=checkpoint["self_1_bias"],
            neighbor_1_weight=checkpoint["neighbor_1_weight"],
            self_2_weight=checkpoint["self_2_weight"],
            self_2_bias=checkpoint["self_2_bias"],
            neighbor_2_weight=checkpoint["neighbor_2_weight"],
            output_weight=checkpoint["output_weight"],
            output_bias=checkpoint["output_bias"],
        )
        return cls(
            model,
            model_id=str(manifest.get("model_id", MODEL_ID)),
            minimum_risk_probability=float(
                manifest.get("minimum_risk_probability", 0.50)
            ),
        )

    def analyze(self, frames: list[TwinFrame]) -> list[CellRiskAnalysis | None]:
        if not frames:
            return []
        if self.model is None:
            return [None] * len(frames)
        features = np.stack([frame_features(frame) for frame in frames], axis=0)
        outputs: list[CellRiskAnalysis] = []
        probability_batches: list[np.ndarray] = []
        for offset in range(0, len(frames), 256):
            logits = self.model.logits(features[offset : offset + 256])
            logits -= logits.max(axis=-1, keepdims=True)
            exponentials = np.exp(logits)
            probability_batches.append(
                exponentials / exponentials.sum(axis=-1, keepdims=True)
            )
        probabilities = np.concatenate(probability_batches, axis=0)
        for cell_probabilities in probabilities:
            non_normal = 1.0 - cell_probabilities[:, 0]
            severity = cell_probabilities @ np.asarray(
                [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
                dtype=np.float32,
            )
            levels = cell_probabilities.argmax(axis=-1).astype(np.int64)
            levels[non_normal < self.minimum_risk_probability] = 0
            source_frame = frames[len(outputs)]
            temperatures = np.asarray(
                source_frame.temperature_decic,
                dtype=np.float32,
            ) / 10.0
            voltages = np.asarray(source_frame.voltage_mv, dtype=np.float32) / 1_000.0
            temperature_salience = np.clip(
                (temperatures - float(np.median(temperatures))) / 40.0,
                0.0,
                1.0,
            )
            voltage_salience = np.clip(
                np.abs(voltages - float(np.median(voltages))) / 1.5,
                0.0,
                1.0,
            )
            measurement_salience = np.maximum.reduce(
                (temperature_salience, voltage_salience)
            )
            hotspot_score = severity + 0.05 * measurement_salience
            all_affected = np.flatnonzero(levels > 0).tolist()
            hotspot = (
                int(max(all_affected, key=hotspot_score.__getitem__))
                if all_affected
                else int(np.argmax(hotspot_score))
            )
            affected = sorted(
                all_affected,
                key=lambda index: (
                    int(levels[index]),
                    float(severity[index]),
                    float(measurement_salience[index]),
                ),
                reverse=True,
            )
            outputs.append(
                CellRiskAnalysis(
                    model_id=self.model_id,
                    risk_score=[round(float(value), 6) for value in non_normal],
                    state_level=[int(value) for value in levels],
                    hotspot_cell_index=hotspot,
                    affected_cell_indices=affected,
                    heat_spread_direction=_direction(
                        hotspot,
                        all_affected,
                        severity,
                    ),
                )
            )
        return outputs

    def enrich(self, frames: list[TwinFrame]) -> list[TwinFrame]:
        analyses = self.analyze(frames)
        enriched: list[TwinFrame] = []
        for frame, analysis in zip(frames, analyses, strict=True):
            if analysis is None:
                enriched.append(frame)
                continue
            cell_levels = [
                max(sensor_level, ai_level)
                for sensor_level, ai_level in zip(
                    frame.state_level,
                    analysis.state_level,
                    strict=True,
                )
            ]
            module_levels = [
                max(cell_levels[offset : offset + CELLS_PER_MODULE])
                for offset in range(0, CELL_COUNT, CELLS_PER_MODULE)
            ]
            hotspot = (
                analysis.hotspot_cell_index
                if analysis.affected_cell_indices
                else frame.hotspot_cell_index
            )
            enriched.append(
                frame.model_copy(
                    update={
                        "state_level": cell_levels,
                        "module_state_level": module_levels,
                        "hotspot_cell_index": hotspot,
                        "hotspot_module_index": hotspot // CELLS_PER_MODULE,
                        "twin_ai_model_id": analysis.model_id,
                        "twin_ai_status": "ready",
                        "cell_ai_risk_score": analysis.risk_score,
                        "cell_ai_state_level": analysis.state_level,
                        "affected_cell_indices": analysis.affected_cell_indices,
                        "heat_spread_direction": analysis.heat_spread_direction,
                    }
                )
            )
        return enriched

    def enrich_one(self, frame: TwinFrame) -> TwinFrame:
        return self.enrich([frame])[0]
