"""stored CSV tables for exact whole-dataset analysis

Revision ID: 0003
Revises: 0002
"""
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "rag_tables",
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("columns", JSON, nullable=False),
        sa.Column("rows", JSON, nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["rag_documents.document_id"], name=op.f("fk_rag_tables_document_id_rag_documents"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("document_id", name=op.f("pk_rag_tables")),
    )
    op.create_index(op.f("ix_rag_tables_user_id"), "rag_tables", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_table("rag_tables")
