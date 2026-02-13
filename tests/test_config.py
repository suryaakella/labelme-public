from config import Settings, settings


def test_settings_defaults():
    s = Settings()
    assert s.postgres_host == "localhost"
    assert s.postgres_port == 5432
    assert s.max_workers == 4
    assert s.dedup_threshold == 0.95


def test_database_url():
    s = Settings()
    assert "postgresql+asyncpg://" in s.database_url
    assert s.postgres_db in s.database_url


def test_singleton():
    assert settings is not None
    assert isinstance(settings, Settings)
