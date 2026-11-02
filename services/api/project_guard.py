"""The by-id half of project visibility (services/project_access.py), applied to every API router.

Any route whose path or query string names a session, frame, object or track answers 404 when that item sits in a members-only
project the caller is not in. It costs nothing when no project is members-only or the caller is an
administrator. The set of hidden projects is left on request.state for listing routes to reuse.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import current_user, db_session
from services.project_access import first_hidden_item, hidden_projects


async def project_guard(request: Request, user=Depends(current_user), db: AsyncSession = Depends(db_session)):
    hidden = await hidden_projects(db, user)
    request.state.hidden_projects = hidden
    if hidden and await first_hidden_item(db, hidden, request.path_params, dict(request.query_params)):
        # The same answer a missing item gets, so a refusal does not confirm the item exists.
        raise HTTPException(404, "not found")


async def caller_hidden_projects(request: Request, user=Depends(current_user),
                                 db: AsyncSession = Depends(db_session)):
    """For a listing route: the projects to leave out, computed once per request."""
    hidden = getattr(request.state, "hidden_projects", None)
    if hidden is None:
        hidden = await hidden_projects(db, user)
        request.state.hidden_projects = hidden
    return hidden


PROJECT_GUARD = [Depends(project_guard)]
