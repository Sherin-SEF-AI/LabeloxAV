"""No request handler does blocking storage or image work on the event loop.

The API is one process with one loop. An object-store read is a blocking HTTP call and a decode is tens
of milliseconds of CPU; written inline in an `async def` handler, either stops every other request until
it returns. Thirty-odd handlers did exactly that. This reads the router sources and fails on the next
one, so the rule is enforced where the code is written instead of found again under load.

A call is fine inside a nested `def` or `lambda`, which is how work is handed to a thread
(services/api/offload.py). It is the call written directly in the coroutine body that blocks.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKING = {"get_bytes", "put_bytes", "get_stream", "ensure_bucket", "put_file", "bucket_reachable",
            "imdecode", "imencode", "imread"}


def _offenders(path: Path) -> list[str]:
    src = path.read_text()
    tree = ast.parse(src)
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in BLOCKING):
            continue
        p = parents.get(node)
        while p is not None and not isinstance(p, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            p = parents.get(p)
        if isinstance(p, ast.AsyncFunctionDef):
            out.append(f"{path.relative_to(ROOT)}:{node.lineno} {node.func.attr}() in async def {p.name}")
    return out


def test_the_scanner_sees_what_it_claims_to(tmp_path):
    f = tmp_path / "x.py"
    f.write_text(
        "import asyncio\n"
        "async def bad(store):\n"
        "    return store.get_bytes('k')\n"
        "async def good(store):\n"
        "    def _load():\n"
        "        return store.get_bytes('k')\n"
        "    return await asyncio.to_thread(_load)\n"
        "async def also_good(store):\n"
        "    return await asyncio.to_thread(store.get_bytes, 'k')\n")
    global ROOT
    saved, ROOT = ROOT, tmp_path
    try:
        found = _offenders(f)
    finally:
        ROOT = saved
    assert len(found) == 1 and "bad" in found[0]


def test_no_handler_blocks_the_event_loop():
    files = sorted((ROOT / "services" / "api" / "routers").glob("*.py")) + [ROOT / "services" / "api" / "main.py"]
    assert len(files) > 50, "the router directory was not found; this test would pass on nothing"
    offenders = [o for f in files for o in _offenders(f)]
    assert offenders == [], ("blocking calls on the event loop; hand them to a thread "
                             "(services/api/offload.py):\n  " + "\n  ".join(offenders))
