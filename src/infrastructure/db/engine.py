"""Async SQLAlchemy engine factory."""
from functools import lru_cache
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine, async_sessionmaker


@lru_cache(maxsize=1)
def get_engine(db_url: str) -> AsyncEngine:
    # psycopg v3 async driver: use `postgresql+psycopg://` (not asyncpg).
    return create_async_engine(db_url, pool_pre_ping=True, future=True)


def session_factory(engine: AsyncEngine):
    return async_sessionmaker(engine, expire_on_commit=False)
