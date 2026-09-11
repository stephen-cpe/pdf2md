"""Async engine + session factory."""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def create_engine(dsn: str) -> AsyncEngine:
    """AsyncPG engine for the given DSN (expects the asyncpg URL form)."""
    return create_async_engine(dsn)


def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Session factory with no autoflush surprises for checkpoint code."""
    return async_sessionmaker(engine, expire_on_commit=False)


__all__ = ["create_engine", "session_factory"]
