"""Keep tests/KNOWN_FAILURES.md honest.

The suite is not green, and the runbook told operators to "compare a run against the baseline" without there
being a baseline to compare against, which makes "is the build broken?" unanswerable mechanically. That file
is now the baseline; this test stops it from rotting into fiction by checking that every test it names still
exists and that every xfail in the tree is accounted for in it.

It deliberately does not run the failing tests. It checks the bookkeeping, which is the part that silently
goes stale."""
from __future__ import annotations

import re
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
DOC = TESTS_DIR / "KNOWN_FAILURES.md"


def _documented_tests() -> set[str]:
    """Every `file.py::test_name` named in the baseline document."""
    return set(re.findall(r"`(test_[\w]+\.py::[\w]+)`", DOC.read_text()))


def _xfailed_tests() -> set[str]:
    """Every test carrying an xfail marker, as file.py::test_name."""
    found: set[str] = set()
    for path in TESTS_DIR.glob("test_*.py"):
        if path.name == Path(__file__).name:
            continue     # this scanner mentions the marker in its own source; do not match on itself
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            if "pytest.mark.xfail" not in line:
                continue
            # the decorated function is the next def, skipping stacked decorators and wrapped arguments
            for follow in lines[i:]:
                m = re.match(r"\s*(?:async\s+)?def\s+(test_\w+)", follow)
                if m:
                    found.add(f"{path.name}::{m.group(1)}")
                    break
    return found


def test_baseline_document_exists_and_is_not_empty():
    assert DOC.exists(), "the recorded baseline is what makes a red run interpretable"
    assert _documented_tests(), "the baseline names no tests, so it cannot be compared against"


def test_every_documented_test_still_exists():
    # A renamed or deleted test left in the baseline means a real regression could hide behind a stale name.
    missing = []
    for ref in sorted(_documented_tests()):
        filename, testname = ref.split("::")
        path = TESTS_DIR / filename
        if not path.exists() or f"def {testname}" not in path.read_text():
            missing.append(ref)
    assert missing == [], f"baseline names tests that no longer exist: {missing}"


def test_every_xfail_is_documented():
    # An xfail is a failure someone decided to tolerate. If it is not written down with a reason and a route
    # to a fix, it is just a hidden failure.
    undocumented = sorted(_xfailed_tests() - _documented_tests())
    assert undocumented == [], f"xfailed but absent from KNOWN_FAILURES.md: {undocumented}"


def test_each_section_states_a_fix():
    # Every category has to say what would make it go away, so the list cannot quietly become permanent.
    text = DOC.read_text()
    sections = [s for s in text.split("\n## ")[1:]]
    assert sections, "no sections found"
    for s in sections:
        assert "**Fix:**" in s, f"section without a stated fix: {s.splitlines()[0]}"
