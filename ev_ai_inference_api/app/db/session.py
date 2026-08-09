from __future__ import annotations

from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def _prepare_database_url(database_url: str) -> tuple[URL, dict[str, object]]:
    """Move asyncpg SSL options out of the URL query.

    SQLAlchemy forwards unknown asyncpg URL query parameters as PostgreSQL
    runtime settings.  Leaving ``ssl=require`` in the URL therefore makes
    asyncpg execute it as a server parameter and PostgreSQL rejects the
    connection.  asyncpg expects SSL through its ``ssl`` connect argument.
    """

    url = make_url(database_url)
    ssl_value = url.query.get("ssl")
    sslmode_value = url.query.get("sslmode")

    if ssl_value is not None and sslmode_value is not None:
        if ssl_value != sslmode_value:
            raise ValueError("ssl and sslmode database URL options must match")

    effective_ssl = ssl_value if ssl_value is not None else sslmode_value
    if effective_ssl is None:
        return url, {}

    url = url.difference_update_query(("ssl", "sslmode"))
    return url, {"ssl": effective_ssl}


def create_database(
    database_url: str,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    url, connect_args = _prepare_database_url(database_url)
    engine_options: dict[str, object] = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
    }
    if connect_args:
        engine_options["connect_args"] = connect_args

    engine = create_async_engine(
        url,
        **engine_options,
    )
    return engine, async_sessionmaker(engine, expire_on_commit=False)
