from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from app.schemas.twins import CELL_COUNT, MODULE_COUNT, TwinSampleRequest


IMAGE_WIDTH = 336
IMAGE_HEIGHT = 297
MODULE_ROWS = 3
MODULE_COLUMNS = 4
CELLS_PER_MODULE = 8
MODULE_CELL_ROWS = 2
MODULE_CELL_COLUMNS = 4


@dataclass(frozen=True)
class ThermalRender:
    image_bytes: bytes
    cell_heat_score: tuple[float, ...]
    module_heat_score: tuple[float, ...]
    hotspot_cell_index: int
    hotspot_module_index: int
    sha256: str


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _cell_centers() -> tuple[tuple[float, float], ...]:
    margin_x = 42.0
    margin_y = 38.0
    cell_width = 14.0
    cell_height = 28.0
    module_gap_x = 12.0
    module_gap_y = 12.0
    centers: list[tuple[float, float]] = []
    for module_row in range(MODULE_ROWS):
        for module_column in range(MODULE_COLUMNS):
            module_x = margin_x + module_column * (
                MODULE_CELL_COLUMNS * cell_width + module_gap_x
            )
            module_y = margin_y + module_row * (
                MODULE_CELL_ROWS * cell_height + module_gap_y
            )
            for cell_row in range(MODULE_CELL_ROWS):
                for cell_column in range(MODULE_CELL_COLUMNS):
                    centers.append(
                        (
                            module_x + (cell_column + 0.5) * cell_width,
                            module_y + (cell_row + 0.5) * cell_height,
                        )
                    )
    if len(centers) != CELL_COUNT:
        raise RuntimeError("thermal cell layout does not contain 96 cells")
    return tuple(centers)


CELL_CENTERS = _cell_centers()


def _cell_grid_position(index: int) -> tuple[int, int]:
    """Return the row/column in the 6x16 physical cell grid."""

    module_row, module_column = divmod(index // CELLS_PER_MODULE, MODULE_COLUMNS)
    cell_offset = index % CELLS_PER_MODULE
    cell_row, cell_column = divmod(cell_offset, MODULE_CELL_COLUMNS)
    return module_row * MODULE_CELL_ROWS + cell_row, module_column * MODULE_CELL_COLUMNS + cell_column


def _cell_index_at(row: int, column: int) -> int | None:
    if not (0 <= row < MODULE_ROWS * MODULE_CELL_ROWS):
        return None
    if not (0 <= column < MODULE_COLUMNS * MODULE_CELL_COLUMNS):
        return None
    module_row, cell_row = divmod(row, MODULE_CELL_ROWS)
    module_column, cell_column = divmod(column, MODULE_CELL_COLUMNS)
    module = module_row * MODULE_COLUMNS + module_column
    return module * CELLS_PER_MODULE + cell_row * MODULE_CELL_COLUMNS + cell_column


def sensor_cell_heat_scores(
    temperatures: list[int] | tuple[int, ...],
    ambient: float,
) -> tuple[float, ...]:
    """Convert the 96 sensor temperatures to continuous 0..1 heat scores."""

    if len(temperatures) != CELL_COUNT:
        raise ValueError("temperature array must contain 96 values")
    return tuple(
        _clamp((temperature / 10.0 - ambient - 5.0) / 75.0)
        for temperature in temperatures
    )


def spread_cell_heat_scores(scores: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    """Apply one-hop orthogonal heat diffusion without copying module colors."""

    if len(scores) != CELL_COUNT:
        raise ValueError("cell heat score array must contain 96 values")
    source = [_clamp(float(score)) for score in scores]
    result = list(source)
    positions = [_cell_grid_position(index) for index in range(CELL_COUNT)]
    for index, score in enumerate(source):
        if score <= 0.0:
            continue
        row, column = positions[index]
        module = index // CELLS_PER_MODULE
        for row_delta, column_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = _cell_index_at(row + row_delta, column + column_delta)
            if neighbor is None:
                continue
            weight = 0.35 if neighbor // CELLS_PER_MODULE == module else 0.20
            result[neighbor] = max(result[neighbor], _clamp(score * weight))
    return tuple(result)


def module_scores_from_cells(scores: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    if len(scores) != CELL_COUNT:
        raise ValueError("cell heat score array must contain 96 values")
    return tuple(
        max(float(score) for score in scores[index : index + CELLS_PER_MODULE])
        for index in range(0, CELL_COUNT, CELLS_PER_MODULE)
    )


def _hotness_from_rgb(rgb: np.ndarray) -> np.ndarray:
    red = rgb[..., 0].astype(np.float32) / 255.0
    green = rgb[..., 1].astype(np.float32) / 255.0
    blue = rgb[..., 2].astype(np.float32) / 255.0
    return np.clip(0.62 * (red - blue) + 0.38 * (red - green) + 0.12, 0.0, 1.0)


def analyze_cell_heat_scores(image_bytes: bytes) -> tuple[tuple[float, ...], int]:
    """Estimate one heat score per fixed 96-cell ROI and its hotspot cell."""

    with Image.open(io.BytesIO(image_bytes)) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("thermal image must decode to RGB")
    hotness = _hotness_from_rgb(rgb)
    height, width = hotness.shape
    scale_x = width / IMAGE_WIDTH
    scale_y = height / IMAGE_HEIGHT
    scores: list[float] = []
    for center_x, center_y in CELL_CENTERS:
        x = int(round(center_x * scale_x))
        y = int(round(center_y * scale_y))
        radius_x = max(2, int(round(5.0 * scale_x)))
        radius_y = max(2, int(round(8.0 * scale_y)))
        region = hotness[
            max(0, y - radius_y) : min(height, y + radius_y + 1),
            max(0, x - radius_x) : min(width, x + radius_x + 1),
        ]
        if not region.size:
            scores.append(0.0)
            continue
        threshold = float(np.quantile(region, 0.75))
        top_pixels = region[region >= threshold]
        scores.append(float(np.clip(top_pixels.mean() if top_pixels.size else region.mean(), 0.0, 1.0)))
    diffused = spread_cell_heat_scores(scores)
    hotspot = max(range(CELL_COUNT), key=diffused.__getitem__)
    return diffused, hotspot


def analyze_module_heat_scores(image_bytes: bytes) -> tuple[tuple[float, ...], int]:
    """Backward-compatible 12-module summary of the 96-cell analysis."""

    cell_scores, hotspot_cell = analyze_cell_heat_scores(image_bytes)
    module_scores = module_scores_from_cells(cell_scores)
    return module_scores, hotspot_cell // CELLS_PER_MODULE


def render_thermal_frame(
    vehicle_id: str,
    sample: TwinSampleRequest,
    *,
    blur_radius: float = 0.55,
) -> ThermalRender:
    """Render a deterministic, synchronized battery thermal image for local demos."""

    ambient = float(sample.ambient_temperature_c or 25.0)
    temperatures = list(sample.temperature_decic)
    seed_bytes = f"{vehicle_id}:{sample.sequence}:{sample.observed_at.isoformat()}".encode()
    seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]
    baseline = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 0.08, dtype=np.float32)
    baseline += rng.normal(0.0, 0.012, baseline.shape).astype(np.float32)
    field = baseline.copy()
    normalized_values = list(sensor_cell_heat_scores(temperatures, ambient))
    for (center_x, center_y), strength in zip(
        CELL_CENTERS, normalized_values, strict=True
    ):
        sigma = 4.5 + 9.5 * strength
        distance = ((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma**2)
        field += (0.20 + 0.90 * strength) * np.exp(-distance)
    field = np.clip(field, 0.0, 1.0)
    red = np.clip(0.08 + 1.75 * field, 0.0, 1.0)
    green = np.clip(np.clip(2.50 * field, 0.0, 1.0) * np.clip(1.60 - 1.30 * field, 0.25, 1.0), 0.025, 1.0)
    blue = np.clip(0.22 + 0.78 * (1.0 - field) ** 1.2, 0.0, 1.0)
    rgb = np.stack((red, green, blue), axis=-1)
    rgb += rng.normal(0.0, 0.009, rgb.shape)
    image = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8))
    if blur_radius > 0.0:
        image = image.filter(ImageFilter.GaussianBlur(float(blur_radius)))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    image_bytes = buffer.getvalue()
    cell_scores = spread_cell_heat_scores(normalized_values)
    module_scores = module_scores_from_cells(cell_scores)
    hotspot_cell = max(range(CELL_COUNT), key=cell_scores.__getitem__)
    hotspot_module = hotspot_cell // CELLS_PER_MODULE
    return ThermalRender(
        image_bytes=image_bytes,
        cell_heat_score=cell_scores,
        module_heat_score=module_scores,
        hotspot_cell_index=hotspot_cell,
        hotspot_module_index=hotspot_module,
        sha256=hashlib.sha256(image_bytes).hexdigest(),
    )
