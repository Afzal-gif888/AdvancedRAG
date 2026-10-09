"""ingestion attempts: per-document lock, progress and vector extent

Revision ID: 0002
Revises: 0001
"""
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "rag_ingestions",
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("attempt_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("stage", sa.String(length=40), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("vector_extent", sa.Integer(), nullable=False),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.String(length=300), nullable=True),
        sa.Column("retryable", sa.Boolean(), nullable=False),
        sa.Column("timings", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=True),
        sa.Column("heartbeat_at", TS, nullable=False),
        sa.Column("started_at", TS, nullable=False),
        sa.Column("finished_at", TS, nullable=True),
        sa.PrimaryKeyConstraint("document_id", name=op.f("pk_rag_ingestions")),
    )
    op.create_index(op.f("ix_rag_ingestions_user_id"), "rag_ingestions", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_table("rag_ingestions")
