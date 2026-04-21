"""initial schema: reports, receipts, traces"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID as PgUUID

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "reports",
        sa.Column("id", PgUUID(as_uuid=True), primary_key=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("prompt", sa.Text, nullable=True),
        sa.Column("input_kind", sa.String(16), nullable=False),
        sa.Column("input_ref", sa.Text, nullable=True),
        sa.Column("receipt_count", sa.Integer, nullable=True),
        sa.Column("total_spend", sa.Numeric(14, 2), nullable=True),
        sa.Column("by_category", JSONB, nullable=True),
        sa.Column("issues", JSONB, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
    )

    op.create_table(
        "receipts",
        sa.Column("id", PgUUID(as_uuid=True), primary_key=True),
        sa.Column("report_id", PgUUID(as_uuid=True),
                  sa.ForeignKey("reports.id", ondelete="CASCADE"), nullable=False),
        sa.Column("seq", sa.Integer, nullable=False),
        sa.Column("source_ref", sa.Text, nullable=False),
        sa.Column("vendor", sa.Text),
        sa.Column("receipt_date", sa.Date),
        sa.Column("receipt_number", sa.Text),
        sa.Column("total", sa.Numeric(14, 2)),
        sa.Column("currency", sa.String(8)),
        sa.Column("category", sa.Text),
        sa.Column("confidence", sa.Numeric(3, 2)),
        sa.Column("notes", sa.Text),
        sa.Column("issues", JSONB),
        sa.Column("raw_ocr", JSONB),
        sa.Column("normalized", JSONB),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("error", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("idx_receipts_report_seq", "receipts", ["report_id", "seq"])

    op.create_table(
        "traces",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("report_id", PgUUID(as_uuid=True),
                  sa.ForeignKey("reports.id", ondelete="CASCADE"), nullable=False),
        sa.Column("receipt_id", PgUUID(as_uuid=True), nullable=True),  # NOT a FK
        sa.Column("seq", sa.Integer, nullable=False),
        sa.Column("event_type", sa.String(32), nullable=False),
        sa.Column("step", sa.String(32)),
        sa.Column("tool", sa.String(64)),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("duration_ms", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("idx_traces_report_seq", "traces", ["report_id", "seq"])
    op.create_index("idx_traces_report_type", "traces", ["report_id", "event_type"])


def downgrade():
    op.drop_index("idx_traces_report_type", table_name="traces")
    op.drop_index("idx_traces_report_seq", table_name="traces")
    op.drop_table("traces")
    op.drop_index("idx_receipts_report_seq", table_name="receipts")
    op.drop_table("receipts")
    op.drop_table("reports")
