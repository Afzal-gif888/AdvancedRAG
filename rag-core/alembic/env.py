import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context

config = context.config
if config.config_file_name is not None and not config.attributes.get("skip_logging_config"):
    fileConfig(config.config_file_name)

from rag_core.db import Base  # noqa: E402

target_metadata = Base.metadata

url = config.attributes.get("database_url")
if not url:
    from rag_core.config import settings

    url = settings.database_url

from advancerag_common.db import async_engine_args  # noqa: E402

# Provider URLs (Neon/Railway: sslmode=require, pooled endpoints) are normalised for asyncpg.
url, connect_args = async_engine_args(url)
config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))


def _run(connection):
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def _online():
    engine = create_async_engine(url, poolclass=pool.NullPool, connect_args=connect_args)
    async with engine.connect() as conn:
        await conn.run_sync(_run)
    await engine.dispose()


if context.is_offline_mode():
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
else:
    asyncio.run(_online())
