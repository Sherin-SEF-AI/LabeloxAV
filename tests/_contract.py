"""Proof that a declared response model loses nothing the handler returned.

`response_model` is a filter. FastAPI validates what a handler returned into the model and sends the model,
so a key the model does not name is dropped from the JSON without an error anywhere, and the first sign is a
blank panel in the web app. Reading a handler and writing a model for it is exactly the kind of work that
gets one conditional key wrong, so every route that declares a model is checked the same way, here:

  1. `call()` sends a real request through the mounted app and records what the handler function returned
     on that same request, before FastAPI filtered it. Not a second call to the handler: the two would see
     different rows the moment a test seeds between them, and a comparison that can differ for a reason
     other than the filter proves nothing about the filter.
  2. `assert_nothing_lost()` requires the body the client received to equal the handler's return value,
     JSON-encoded. Equal, not "has the same keys": that also catches a key the model added (an optional
     field arriving as null where the handler left it out) and a value it changed.
  3. It then walks the handler's return value against the declared model and requires every key, at every
     depth, to be a declared field. The models allow extra keys through (services/api/deps.py, ApiOut), so
     step 2 alone would pass for a key nobody declared; this is what makes "the key survived" and "the key
     is in the generated types" the same claim.
  4. It refuses a vacuous pass: a declared list of models that came back empty checked no model at all, so
     a case has to say so on purpose (`empty_ok`) or seed a row.

The recording works by wrapping `route.dependant.call`, which is the function FastAPI invokes with the
solved dependencies. It is the only place the pre-filter value exists; the handler's own name in its module
is not what the router holds.
"""

from __future__ import annotations

import inspect
import types
import typing
from dataclasses import dataclass
from typing import Any

from fastapi.encoders import jsonable_encoder
from fastapi.routing import APIRoute
from pydantic import BaseModel
from starlette.routing import Match

_UNSET: Any = object()


@dataclass
class Exchange:
    """One request: the route it reached, what its handler returned, and what the client was sent."""

    method: str
    path: str
    route: APIRoute
    raw: Any
    response: Any

    @property
    def operation(self) -> str:
        """"GET /api/frames/{frame_id}": the key the generated clients use for this route."""
        return f"{self.method} {self.route.path}"

    @property
    def sent(self) -> Any:
        return self.response.json()


def route_for(app, method: str, path: str) -> APIRoute:
    """The route this request will reach: the first full match in registration order, as Starlette picks."""
    scope = {"type": "http", "method": method.upper(), "path": path.split("?", 1)[0], "root_path": ""}
    for route in app.routes:
        if isinstance(route, APIRoute) and route.matches(scope)[0] == Match.FULL:
            return route
    raise AssertionError(f"no route matches {method} {path}")


def call(client, method: str, path: str, **kwargs) -> Exchange:
    """Send one request and capture the handler's own return value alongside the response."""
    method = method.upper()
    route = route_for(client.app, method, path)
    original = route.dependant.call
    # FastAPI decided at registration whether to await this or run it in a thread, from the original
    # function. The stand-in below is a coroutine, so it is only a faithful stand-in for one.
    assert inspect.iscoroutinefunction(original), f"{route.path}: handler is not async; extend call()"
    seen = {"raw": _UNSET}

    async def recording(**values):
        seen["raw"] = await original(**values)
        return seen["raw"]

    route.dependant.call = recording
    try:
        response = client.request(method, path, **kwargs)
    finally:
        route.dependant.call = original
    return Exchange(method=method, path=path, route=route, raw=seen["raw"], response=response)


def _is_model(tp: Any) -> bool:
    return inspect.isclass(tp) and issubclass(tp, BaseModel)


def _members(tp: Any) -> list[Any]:
    """A union's alternatives without None; any other annotation as a one-element list."""
    if typing.get_origin(tp) in (typing.Union, types.UnionType):
        return [a for a in typing.get_args(tp) if a is not type(None)]
    return [tp]


def walk(tp: Any, value: Any, path: str, undeclared: list[str], vacuous: list[str]) -> None:
    """Compare a handler's value with the annotation that declares it.

    Appends to `undeclared` the path of every key the value carries and its model does not name, and to
    `vacuous` the path of every declared list of models that is empty (nothing in it was checked).
    """
    if value is None:
        return
    for member in _members(tp):
        origin = typing.get_origin(member)
        if _is_model(member) and isinstance(value, dict):
            fields = member.model_fields
            for key in value:
                if key not in fields:
                    undeclared.append(f"{path}.{key}  (not a field of {member.__name__})")
                else:
                    walk(fields[key].annotation, value[key], f"{path}.{key}", undeclared, vacuous)
        elif origin is list and isinstance(value, list | tuple):
            (item,) = typing.get_args(member) or (Any,)
            if not value and any(_is_model(m) for m in _members(item)):
                vacuous.append(path)
            for i, v in enumerate(value):
                walk(item, v, f"{path}[{i}]", undeclared, vacuous)
        elif origin is dict and isinstance(value, dict):
            args = typing.get_args(member)
            for k, v in value.items():
                walk(args[1] if len(args) == 2 else Any, v, f"{path}[{k!r}]", undeclared, vacuous)


def models_in(tp: Any) -> set[type[BaseModel]]:
    """Every model class an annotation reaches, through lists, dicts, unions and nested fields."""
    found: set[type[BaseModel]] = set()

    def visit(t: Any) -> None:
        for member in _members(t):
            if _is_model(member):
                if member not in found:
                    found.add(member)
                    for f in member.model_fields.values():
                        visit(f.annotation)
            else:
                for arg in typing.get_args(member):
                    visit(arg)

    visit(tp)
    return found


def assert_nothing_lost(ex: Exchange, *, empty_ok: tuple[str, ...] = ()) -> Any:
    """The response is the handler's return value, whole, and every key of it is declared. Returns the
    body so a case can go on to assert what it was seeded to show."""
    assert ex.response.status_code == 200, f"{ex.method} {ex.path} -> {ex.response.status_code}: {ex.response.text[:400]}"
    assert ex.raw is not _UNSET, f"{ex.operation}: the handler was never reached"
    model = ex.route.response_model
    assert model is not None, f"{ex.operation} declares no response model"

    expected = jsonable_encoder(ex.raw)
    assert ex.sent == expected, (
        f"{ex.operation}: the body sent differs from what the handler returned.\n"
        f"  handler: {expected!r}\n  sent:    {ex.sent!r}")

    undeclared: list[str] = []
    vacuous: list[str] = []
    walk(model, ex.raw, "$", undeclared, vacuous)
    assert not undeclared, (
        f"{ex.operation}: the handler returns keys its response model does not declare, so the generated "
        f"clients do not know them:\n  " + "\n  ".join(undeclared))
    unexplained = [p for p in vacuous if p not in empty_ok]
    assert not unexplained, (
        f"{ex.operation}: nothing was checked at {unexplained}: the list is empty, so no row was compared "
        f"with its model. Seed a row, or name the path in empty_ok if empty is the point.")
    return ex.sent
