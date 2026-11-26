"""Tables for the agents annotation tools (migration 0129_tools_agents).

Define classes on `db.models.Base` here and list them in `__all__`; db/models.py re-exports them.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from db.models import Base

__all__: list[str] = ["AutopilotConfig", "AutopilotNight"]

DEFAULT_STEPS = {"autolabel": True, "vlm_check": True, "error_sweep": True, "auto_accept": True}


class AutopilotConfig(Base):
    """The overnight autopilot's settings: one row, id 1, created on first read."""

    __tablename__ = "autopilot_config"

    id: Mapped[int] = mapped_column(SmallInteger, primary_key=True, default=1)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default=sql_text("false"))
    killed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default=sql_text("false"))
    killed_reason: Mapped[str | None] = mapped_column(Text)
    killed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    gpu_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=240, server_default="240")
    max_sessions: Mapped[int] = mapped_column(Integer, nullable=False, default=20, server_default="20")
    vlm_max_checks: Mapped[int] = mapped_column(Integer, nullable=False, default=300, server_default="300")
    earliest_start_ist: Mapped[str] = mapped_column(String(5), nullable=False, default="20:00",
                                                    server_default="20:00")
    steps: Mapped[dict] = mapped_column(JSONB, nullable=False, default=lambda: dict(DEFAULT_STEPS))
    updated_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("app_user.user_id", ondelete="SET NULL"))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(),
                                                 onupdate=func.now())


class AutopilotNight(Base):
    """One night the autopilot ran, or a dry run of one: the plan, the morning report, the GPU time used."""

    __tablename__ = "autopilot_night"

    night_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    night_date: Mapped[date] = mapped_column(Date, nullable=False)
    dry_run: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default=sql_text("false"))
    # planned | running | done | stopped | error
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="planned", server_default="planned")
    trigger: Mapped[str] = mapped_column(String(16), nullable=False, default="scheduler", server_default="scheduler")
    tool_job_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    agent_run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    settings: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    plan: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    report: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    gpu_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default="0")
    stopped_reason: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("app_user.user_id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_autopilot_night_created", "created_at"),
        Index("ix_autopilot_night_created_by", "created_by"),
        Index("uq_autopilot_night_real", "night_date", unique=True, postgresql_where=sql_text("dry_run = false")),
    )
