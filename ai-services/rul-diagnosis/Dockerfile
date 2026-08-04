FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# fire_risk_model.joblib, rul_model_B_no_cycle.joblib 는 저장소에 포함되지 않음.
# README.md 안내대로 배포 전 이 디렉터리에 두거나 볼륨 마운트할 것.

EXPOSE 8000
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
