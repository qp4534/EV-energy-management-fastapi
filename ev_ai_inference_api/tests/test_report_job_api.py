from uuid import UUID

from fastapi.testclient import TestClient

from app.reporting.main import create_report_job_app


class FakeQueue:
    def __init__(self):
        self.calls = []

    async def enqueue_anomaly(self, anomaly_id):
        self.calls.append(("ANOMALY", anomaly_id))
        return UUID("11111111-1111-1111-1111-111111111111")

    async def enqueue_monthly(self, car_id, target_month):
        self.calls.append(("MONTHLY", car_id, target_month))
        return UUID("22222222-2222-2222-2222-222222222222")


def test_monthly_job_api_accepts_spring_contract() -> None:
    queue = FakeQueue()
    app = create_report_job_app(queue)
    with TestClient(app) as client:
        response = client.post(
            "/internal/v1/report-jobs/monthly",
            json={
                "carId": "33333333-3333-3333-3333-333333333333",
                "targetMonth": "2026-07",
            },
        )

    assert response.status_code == 202
    assert response.json() == {
        "jobId": "22222222-2222-2222-2222-222222222222",
        "status": "PENDING",
    }
    assert queue.calls[0][2].isoformat() == "2026-07-01"
