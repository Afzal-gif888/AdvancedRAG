"""refresh-token grace window (previous hash + rotation time)

Revision ID: 0002
Revises: 0001
"""
import sqlalchemy as sa

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("sessions") as batch:
        batch.add_column(sa.Column("previous_refresh_token_hash", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("rotated_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("sessions") as batch:
        batch.drop_column("rotated_at")
        batch.drop_column("previous_refresh_token_hash")
