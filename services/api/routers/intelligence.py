"""Scenario mining endpoints: trigger mining, list/rank scenarios, NL search, detail."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Scenario
from db.models import Session as DbSession
from services.api.deps import db_session
from services.intelligence.nlsearch import _scenario_dict, search_scenarios
from services.intelligence.run import mine_session

router = APIRouter()


@router.post("/mine")
async def mine(payload: dict):
    session_id = payload.get("session_id")
    if not session_id:
        raise HTTPException(400, "session_id required")
    return await mine_session(UUID(session_id))


@router.get("/scenarios")
async def list_scenarios(
    db: AsyncSession = Depends(db_session),
    session_id: str | None = None,
    type: str | None = None,
    city: str | None = None,
    limit: int = 100,
):
    limit = min(max(limit, 1), 1000)
    stmt = (
        select(Scenario, DbSession.city, DbSession.vehicle_id)
        .join(DbSession, Scenario.session_id == DbSession.session_id)
        .order_by(Scenario.criticality.desc())
        .limit(limit)
    )
    if session_id:
        stmt = stmt.where(Scenario.session_id == UUID(session_id))
    if type:
        stmt = stmt.where(Scenario.type == type)
    if city:
        stmt = stmt.where(DbSession.city == city)
    rows = (await db.execute(stmt)).all()
    return [_scenario_dict(s, c, v) for s, c, v in rows]


@router.get("/scenarios/search")
async def scenarios_search(
    db: AsyncSession = Depends(db_session),
    q: str = Query(...),
    city: str | None = None,
    session_id: str | None = None,
    limit: int = 100,
    semantic: bool = False,
):
    results = await search_scenarios(db, q, city=city, session_id=session_id, limit=limit, semantic=semantic)
    return {"query": q, "count": len(results), "results": results, "semantic": semantic}


@router.get("/scenarios/{scenario_id}")
async def scenario_detail(scenario_id: UUID, db: AsyncSession = Depends(db_session)):
    scn = await db.get(Scenario, scenario_id)
    if scn is None:
        raise HTTPException(404, "scenario not found")
    sess = await db.get(DbSession, scn.session_id)
    return _scenario_dict(scn, sess.city if sess else None, sess.vehicle_id if sess else None)
