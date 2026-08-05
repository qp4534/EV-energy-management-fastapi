from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.schemas.current_stage import CurrentStageResponse, SampleRequest


ALERT_RISK_TEXT = {
    "caution": "주의",
    "warning": "경고",
    "emergency": "긴급",
}
ALERT_RISK_NUMBER = {
    "normal": 0,
    "caution": 1,
    "warning": 2,
    "emergency": 3,
}


class AnomalyPersistence:
    """Writes only abnormal current-stage results to the shared ERD tables."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def persist_if_anomalous(
        self,
        car_id: str,
        payload: SampleRequest,
        inference: CurrentStageResponse,
    ) -> str | None:
        alert = inference.final_safety_alert
        if alert not in ALERT_RISK_TEXT:
            return None

        try:
            UUID(car_id)
        except ValueError as exc:
            raise ValueError(
                "vehicle_id must be the CAR.car_id UUID when persistence is enabled"
            ) from exc

        observed_at = payload.observed_at or datetime.now(timezone.utc)
        ml_risk_level = (
            ALERT_RISK_NUMBER[inference.ml_pattern_stage]
            if inference.ml_pattern_stage is not None
            else None
        )
        raw_metrics = json.dumps(
            {
                "request": payload.model_dump(mode="json", exclude_none=True),
                "inference": inference.model_dump(mode="json", exclude_none=True),
            },
            ensure_ascii=False,
        )
        model_input = json.dumps(
            {
                "voltage_v": payload.voltage_v,
                "temp_mean_c": payload.temp_mean_c,
                "temp_max_c": payload.temp_max_c,
                "temp_delta_c": payload.temp_delta_c,
                "temp_saturation_fraction": payload.temp_saturation_fraction,
                "temp_saturation_all": payload.temp_saturation_all,
            },
            ensure_ascii=False,
        )

        async with self._sessions.begin() as session:
            anomaly_id = await session.scalar(
                text(
                    '''
                    INSERT INTO public."ANOMALY_LOGS"
                        (abnormal_type, source_type, trigger_value, detected_at,
                         risk_level, car_id, session_id)
                    VALUES
                        (:abnormal_type, :source_type, :trigger_value, :detected_at,
                         :risk_level, CAST(:car_id AS uuid), CAST(:session_id AS uuid))
                    RETURNING anomaly_id
                    '''
                ),
                {
                    "abnormal_type": "BMS_SAFETY_ALERT",
                    "source_type": "FASTAPI_BMS",
                    "trigger_value": f"final_safety_alert={alert}",
                    "detected_at": observed_at,
                    "risk_level": ALERT_RISK_TEXT[alert],
                    "car_id": car_id,
                    "session_id": str(payload.session_id) if payload.session_id else None,
                },
            )
            await session.execute(
                text(
                    '''
                    INSERT INTO public."TWIN_FRAMES"
                        (observed_at, hotspot_cell_index, hotspot_connector_index,
                         ml_risk_level, physics_risk_level, final_risk_level,
                         image_risk_level, image_confidence, raw_metrics, model_input,
                         anomaly_id, car_id, session_id, source_image_ref)
                    VALUES
                        (:observed_at, :hotspot_cell_index, :hotspot_connector_index,
                         :ml_risk_level, :physics_risk_level, :final_risk_level,
                         :image_risk_level, :image_confidence, CAST(:raw_metrics AS jsonb),
                         CAST(:model_input AS jsonb), :anomaly_id, CAST(:car_id AS uuid),
                         CAST(:session_id AS uuid), :source_image_ref)
                    '''
                ),
                {
                    "observed_at": observed_at,
                    "hotspot_cell_index": payload.hotspot_cell_index,
                    "hotspot_connector_index": payload.hotspot_connector_index,
                    "ml_risk_level": ml_risk_level,
                    "physics_risk_level": ALERT_RISK_NUMBER[inference.physical_rule_level],
                    "final_risk_level": ALERT_RISK_NUMBER[alert],
                    "image_risk_level": payload.image_risk_level,
                    "image_confidence": payload.image_confidence,
                    "raw_metrics": raw_metrics,
                    "model_input": model_input,
                    "anomaly_id": anomaly_id,
                    "car_id": car_id,
                    "session_id": str(payload.session_id) if payload.session_id else None,
                    "source_image_ref": payload.source_image_ref,
                },
            )
        return str(anomaly_id)
