from app.db import session as db_session


def test_create_database_moves_ssl_query_to_connect_args(monkeypatch):
    calls = {}
    fake_engine = object()
    fake_sessions = object()

    def fake_create_async_engine(url, **kwargs):
        calls["url"] = url
        calls["engine_kwargs"] = kwargs
        return fake_engine

    def fake_async_sessionmaker(engine, **kwargs):
        calls["session_engine"] = engine
        calls["session_kwargs"] = kwargs
        return fake_sessions

    monkeypatch.setattr(db_session, "create_async_engine", fake_create_async_engine)
    monkeypatch.setattr(db_session, "async_sessionmaker", fake_async_sessionmaker)

    engine, sessions = db_session.create_database(
        "postgresql+asyncpg://user:password@example.com:5432/ev_database"
        "?ssl=require&application_name=twin"
    )

    assert engine is fake_engine
    assert sessions is fake_sessions
    assert calls["url"].drivername == "postgresql+asyncpg"
    assert dict(calls["url"].query) == {"application_name": "twin"}
    assert calls["engine_kwargs"] == {
        "connect_args": {"ssl": "require"},
        "pool_pre_ping": True,
        "pool_recycle": 300,
    }
    assert calls["session_engine"] is fake_engine
    assert calls["session_kwargs"] == {"expire_on_commit": False}


def test_create_database_leaves_non_ssl_urls_unchanged(monkeypatch):
    calls = {}
    fake_engine = object()

    def fake_create_async_engine(url, **kwargs):
        calls["url"] = url
        calls["engine_kwargs"] = kwargs
        return fake_engine

    monkeypatch.setattr(db_session, "create_async_engine", fake_create_async_engine)
    monkeypatch.setattr(db_session, "async_sessionmaker", lambda *args, **kwargs: object())

    db_session.create_database("postgresql+asyncpg://user:password@example.com/db")

    assert calls["url"].drivername == "postgresql+asyncpg"
    assert calls["url"].username == "user"
    assert calls["url"].password == "password"
    assert calls["url"].host == "example.com"
    assert calls["url"].database == "db"
    assert not calls["url"].query
    assert calls["engine_kwargs"] == {
        "pool_pre_ping": True,
        "pool_recycle": 300,
    }
