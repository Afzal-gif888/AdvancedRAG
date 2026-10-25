"""extraction report, chunk structure metadata and pipeline info (additive, backward compatible)

Revision ID: 0004
Revises: 0003

Every column is nullable, so existing rows and running older code keep working. Nothing is dropped or rewritten.
"""
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column("rag_documents", sa.Column("extraction", JSON, nullable=True))
    op.add_column("rag_documents", sa.Column("pipeline_info", JSON, nullable=True))
    op.add_column("rag_chunks", sa.Column("section", sa.Text(), nullable=True))
    op.add_column("rag_chunks", sa.Column("extraction_method", sa.String(length=16), nullable=True))
    op.add_column("rag_ingestions", sa.Column("extraction", JSON, nullable=True))


def downgrade() -> None:
    op.drop_column("rag_ingestions", "extraction")
    op.drop_column("rag_chunks", "extraction_method")
    op.drop_column("rag_chunks", "section")
    op.drop_column("rag_documents", "pipeline_info")
    op.drop_column("rag_documents", "extraction")
