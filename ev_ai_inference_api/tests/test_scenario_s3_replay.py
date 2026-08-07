from __future__ import annotations

import gzip
import io
import json
from datetime import datetime, timezone

import pytest

from app.scenario_catalog import SCENARIO_BY_ID
from app.scenario_generator import generate_scenario_frames
from app.scenario_replay import (
    load_compact_datasets_from_s3,
    load_datasets_from_s3,
)


START = datetime(2026, 8, 7, tzinfo=timezone.utc)


class NoSuchKey(Exception):
    pass


class FakeS3:
    exceptions = type("Exceptions", (), {"NoSuchKey": NoSuchKey})

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects

    def get_object(self, Bucket: str, Key: str):
        if Key not in self.objects:
            raise NoSuchKey(Key)
        return {"Body": io.BytesIO(self.objects[Key])}


def _gzip_frames(frames) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb") as handle:
        for frame in frames:
            handle.write((frame.model_dump_json() + "\n").encode("utf-8"))
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_load_datasets_from_s3_uses_manifest() -> None:
    normal = await generate_scenario_frames(
        SCENARIO_BY_ID["normal"],
        frame_count=3,
        start_at=START,
    )
    connector = await generate_scenario_frames(
        SCENARIO_BY_ID["connector_local_overheat"],
        frame_count=3,
        start_at=START,
    )
    prefix = "digital-twin/scenarios"
    objects = {
        f"{prefix}/manifest.json": json.dumps(
            {
                "scenarios": [
                    {
                        "scenario_id": "normal",
                        "frames_key": f"{prefix}/normal/frames.jsonl.gz",
                    },
                    {
                        "scenario_id": "connector_local_overheat",
                        "frames_key": (
                            f"{prefix}/connector_local_overheat/"
                            "frames.jsonl.gz"
                        ),
                    },
                ]
            }
        ).encode("utf-8"),
        f"{prefix}/normal/frames.jsonl.gz": _gzip_frames(normal),
        (
            f"{prefix}/connector_local_overheat/frames.jsonl.gz"
        ): _gzip_frames(connector),
    }
    datasets = load_datasets_from_s3(
        "ev-platform-thermal-data",
        prefix,
        s3_client=FakeS3(objects),
    )
    assert set(datasets) == {"normal", "connector_local_overheat"}
    assert len(datasets["normal"]) == 3
    assert len(datasets["connector_local_overheat"]) == 3


@pytest.mark.asyncio
async def test_load_datasets_from_s3_falls_back_without_manifest() -> None:
    normal = await generate_scenario_frames(
        SCENARIO_BY_ID["normal"],
        frame_count=2,
        start_at=START,
    )
    prefix = "digital-twin/scenarios"
    objects = {
        f"{prefix}/normal/frames.jsonl.gz": _gzip_frames(normal),
    }
    datasets = load_datasets_from_s3(
        "ev-platform-thermal-data",
        prefix,
        s3_client=FakeS3(objects),
    )
    assert datasets["normal"]


@pytest.mark.asyncio
async def test_load_compact_datasets_from_s3() -> None:
    normal = await generate_scenario_frames(
        SCENARIO_BY_ID["normal"],
        frame_count=3,
        start_at=START,
    )
    prefix = "digital-twin/scenarios"
    objects = {
        f"{prefix}/manifest.json": json.dumps(
            {
                "scenarios": [
                    {
                        "scenario_id": "normal",
                        "frames_key": f"{prefix}/normal/frames.jsonl.gz",
                        "frame_count": 3,
                    }
                ]
            }
        ).encode("utf-8"),
        f"{prefix}/normal/frames.jsonl.gz": _gzip_frames(normal),
    }
    datasets = load_compact_datasets_from_s3(
        "ev-platform-thermal-data",
        prefix,
        s3_client=FakeS3(objects),
    )
    compact = datasets["normal"]
    assert compact.frame_count == 3
    frame = compact.frame_at(
        2,
        vehicle_id="car-1",
        observed_at=START,
        sequence=2,
    )
    assert frame.vehicle_id == "car-1"
    assert frame.temperature_decic == normal[2].temperature_decic
    assert frame.final_risk_level == normal[2].final_risk_level
