"""rag core schema

Revision ID: 0001
Revises:
"""
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "rag_documents",
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("mime_type", sa.String(length=100), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("max_chunk_count", sa.Integer(), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("embedding_model", sa.String(length=100), nullable=False),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=False),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("updated_at", TS, nullable=False),
        sa.PrimaryKeyConstraint("document_id", name=op.f("pk_rag_documents")),
    )
    op.create_index(op.f("ix_rag_documents_user_id"), "rag_documents", ["user_id"], unique=False)

    op.create_table(
        "rag_chunks",
        sa.Column("id", sa.String(length=80), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("page", sa.Integer(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=True),
        sa.Column("created_at", TS, nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["rag_documents.document_id"], name=op.f("fk_rag_chunks_document_id_rag_documents"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_rag_chunks")),
    )
    op.create_index("ix_rag_chunks_user_id_document_id", "rag_chunks", ["user_id", "document_id"], unique=False)
    if op.get_bind().dialect.name == "postgresql":
        # Keyword half of hybrid retrieval (matches the expression used in pipeline._keyword).
        op.execute("CREATE INDEX ix_rag_chunks_text_fts ON rag_chunks USING GIN (to_tsvector('english', text))")


def downgrade() -> None:
    op.drop_table("rag_chunks")
    op.drop_table("rag_documents")
