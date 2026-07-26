"""India number-plate format: parse, validate, and classify an (OCR'd) plate string.

Pure logic, no model - the fully testable India-specific core of ANPR. Indian registration marks (the
Motor Vehicles Act format) read: a 2-letter state/UT code, a 1-2 digit RTO district code, a 1-3 letter
series, and a 1-4 digit number, e.g. "KA 01 AB 1234". The 2021 Bharat (BH) series reads
"YY BH NNNN LL", e.g. "22 BH 1234 AA". Diplomatic marks read "<n> CD|CC|UN <n>".

We normalise the raw string (uppercase, strip separators), match it against these formats, and validate the
state code against the real RTO list. plate_type distinguishes standard / bh_series / diplomatic / invalid.
Commercial-vs-private is a plate-colour signal (yellow vs white), not derivable from text, so it is not
inferred here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# The real RTO state/UT codes, including the still-seen pre-reorganisation codes (OR->OD Odisha,
# UA->UK Uttarakhand). Sources: MoRTH RTO code list.
STATE_CODES: frozenset[str] = frozenset({
    # states
    "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH", "KA", "KL", "MP", "MH", "MN", "ML",
    "MZ", "NL", "OD", "OR", "PB", "RJ", "SK", "TN", "TS", "TR", "UP", "UK", "UA", "WB",
    # union territories
    "AN", "CH", "DD", "DL", "DN", "LD", "PY", "LA", "JK",
})

_STANDARD = re.compile(r"^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})$")
_BH_SERIES = re.compile(r"^(\d{2})BH(\d{4})([A-Z]{1,2})$")
_DIPLOMATIC = re.compile(r"^(\d{1,3})(CD|CC|UN)(\d{1,4})$")


@dataclass(frozen=True)
class PlateParse:
    raw: str
    normalized: str
    valid: bool
    plate_type: str                    # standard | bh_series | diplomatic | invalid
    state_code: str | None = None
    rto_district: str | None = None
    series: str | None = None
    number: str | None = None
    format_confidence: float = 0.0     # [0,1]: format match strength + state-code validity


def normalize_plate(text: str) -> str:
    """Uppercase and strip everything that is not a letter or digit (spaces, hyphens, dots, the IND band)."""
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def is_valid_state(code: str | None) -> bool:
    return bool(code) and code in STATE_CODES


def parse_plate(text: str) -> PlateParse:
    """Parse a raw plate string into its structured fields with a format confidence.

    A standard mark with a known state code scores 1.0; a well-formed mark with an unknown state code scores
    0.5 (the format is right but the state is not real - likely an OCR error or a fake); anything that matches
    no format is invalid at 0.0, with a best-effort state code if its first two characters are a real code.
    """
    norm = normalize_plate(text)

    m = _STANDARD.match(norm)
    if m:
        state, district, series, number = m.groups()
        known = is_valid_state(state)
        # A 4-digit number is the canonical form; fewer digits is accepted but slightly less certain.
        conf = 1.0 if known else 0.5
        if known and len(number) < 4:
            conf = 0.85
        return PlateParse(raw=text, normalized=norm, valid=known, plate_type="standard",
                          state_code=state, rto_district=district, series=series, number=number,
                          format_confidence=conf)

    m = _BH_SERIES.match(norm)
    if m:
        year, number, series = m.groups()
        return PlateParse(raw=text, normalized=norm, valid=True, plate_type="bh_series",
                          state_code="BH", rto_district=year, series=series, number=number,
                          format_confidence=1.0)

    m = _DIPLOMATIC.match(norm)
    if m:
        code, corps, number = m.groups()
        return PlateParse(raw=text, normalized=norm, valid=True, plate_type="diplomatic",
                          state_code=corps, rto_district=code, series=None, number=number,
                          format_confidence=0.9)

    # No format matched: invalid, but surface a state code if the prefix is a real one (partial read).
    prefix = norm[:2]
    return PlateParse(raw=text, normalized=norm, valid=False, plate_type="invalid",
                      state_code=prefix if is_valid_state(prefix) else None,
                      format_confidence=0.0)
