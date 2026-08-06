from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, datetime, timezone

from app.ai.config import AISettings
from app.ai.deepseek import DeepSeekClient
from app.db.session import create_database
from app.rag.embedding import SentenceTransformerEmbedder
from app.rag.repository import PostgresRagRepository

from .repository import PostgresReportRepository, ReportJob
from .service import ReportGenerationService


LOGGER = logging.getLogger("ev-ai-report-worker")


def previous_month(now: datetime) -> date:
    first = now.date().replace(day=1)
    if first.month == 1:
        return date(first.year - 1, 12, 1)
    return date(first.year, first.month - 1, 1)


async def process_once(
    repository: PostgresReportRepository,
    service: ReportGenerationService,
    *,
    max_retries: int,
) -> bool:
    job = await repository.claim_next()
    if job is None:
        return False
    try:
        title, report = await service.generate(job)
        report_id = await repository.save_report(job, title, report)
        await repository.mark_completed(job.job_id, report_id)
        LOGGER.info("completed report job %s as %s", job.job_key, report_id)
    except Exception as exc:
        LOGGER.exception("report job failed: %s", job.job_key)
        await repository.mark_failed(job, exc, max_retries=max_retries)
    return True


async def run() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = AISettings.load()
    engine, sessions = create_database(settings.database_url)
    repository = PostgresReportRepository(sessions)
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
    )
    rag = PostgresRagRepository(sessions, embedder, settings)
    deepseek = DeepSeekClient(settings)
    service = ReportGenerationService(repository, rag, deepseek)
    try:
        next_monthly_check = 0.0
        while True:
            if time.monotonic() >= next_monthly_check:
                try:
                    recovered = await repository.requeue_stale_running()
                    count = await repository.enqueue_monthly_for_all(
                        previous_month(datetime.now(timezone.utc))
                    )
                    LOGGER.info(
                        "ensured previous-month jobs for %d vehicles; recovered %d stale jobs",
                        count,
                        recovered,
                    )
                except Exception:
                    LOGGER.exception("could not enqueue previous-month report jobs")
                next_monthly_check = time.monotonic() + 3_600
            processed = await process_once(
                repository,
                service,
                max_retries=settings.report_worker_max_retries,
            )
            if not processed:
                await asyncio.sleep(settings.report_worker_poll_seconds)
    finally:
        await deepseek.close()
        await engine.dispose()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
