"""add diagram columns to images

Revision ID: b84dc46ed6a2
Revises: a73cb35dc5f1
Create Date: 2026-09-11 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b84dc46ed6a2"
down_revision: Union[str, Sequence[str], None] = "a73cb35dc5f1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("images", sa.Column("mermaid", sa.Text(), nullable=True))
    op.add_column("images", sa.Column("diagram_type", sa.String(length=64), nullable=True))
    op.add_column("images", sa.Column("conversion_status", sa.String(length=32), nullable=True))
    op.add_column("images", sa.Column("confidence", sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("images", "confidence")
    op.drop_column("images", "conversion_status")
    op.drop_column("images", "diagram_type")
    op.drop_column("images", "mermaid")
