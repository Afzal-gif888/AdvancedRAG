"""durable storage reference, extraction report, document versions and database object storage (additive)

Revision ID: 0005
Revises: 0004

Backward compatible: every new documents column is nullable or has a server default, so existing rows keep
meaning exactly what they meant ('local' storage, version 1, no extraction report, own logical document).
Nothing is dropped or rewritten. Rollback: `alembic downgrade 0004` (drops only what this revision added).
"""
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

TS = sa.DateTime(timezone=True)
JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column("documents", sa.Column("storage_backend", sa.String(length=16), nullable=False, server_default="local"))
    op.add_column("documents", sa.Column("extraction", JSON, nullable=True))
    op.add_column("documents", sa.Column("logical_id", sa.Uuid(), nullable=True))
    op.add_column("documents", sa.Column("version", sa.Integer(), nullable=False, server_default="1"))
    op.add_column("documents", sa.Column("replaces_id", sa.Uuid(), nullable=True))
    op.add_column("documents", sa.Column("superseded_at", TS, nullable=True))
    op.create_index(op.f("ix_documents_logical_id"), "documents", ["logical_id"], unique=False)
    op.create_index("ix_documents_user_id_checksum_sha256", "documents", ["user_id", "checksum_sha256"], unique=False)

    op.create_table(
        "stored_objects",
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("part_count", sa.Integer(), nullable=False),
        sa.Column("created_at", TS, nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name=op.f("fk_stored_objects_user_id_users"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("key", name=op.f("pk_stored_objects")),
    )
    op.create_index(op.f("ix_stored_objects_user_id"), "stored_objects", ["user_id"], unique=False)
    op.create_table(
        "stored_object_parts",
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("idx", sa.Integer(), nullable=False),
        sa.Column("data", sa.LargeBinary(), nullable=False),
        sa.ForeignKeyConstraint(["key"], ["stored_objects.key"], name=op.f("fk_stored_object_parts_key_stored_objects"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("key", "idx", name=op.f("pk_stored_object_parts")),
    )


def downgrade() -> None:
    op.drop_table("stored_object_parts")
    op.drop_index(op.f("ix_stored_objects_user_id"), table_name="stored_objects")
    op.drop_table("stored_objects")
    op.drop_index("ix_documents_user_id_checksum_sha256", table_name="documents")
    op.drop_index(op.f("ix_documents_logical_id"), table_name="documents")
    for col in ("superseded_at", "replaces_id", "version", "logical_id", "extraction", "storage_backend"):
        op.drop_column("documents", col)
