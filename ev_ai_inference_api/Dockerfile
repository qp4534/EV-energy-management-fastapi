FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_BUNDLE_DIR=/app/model_bundles/current_stage_v1/ev_battery_safety_inference_v1 \
    REPORT_JOBS_ENABLED=true \
    EMBEDDED_AI_ENABLED=true \
    REPORT_WORKER_ENABLED=true

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY requirements.txt requirements-ai.txt ./
RUN pip install --no-cache-dir --requirement requirements-ai.txt

# The same image is used by the API, Alembic migration, and persistence worker.
# Runtime commands are overridden per service in docker-compose.yml.
COPY --chown=app:app . .

USER app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
