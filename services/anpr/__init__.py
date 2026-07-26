"""ANPR-India: automatic number-plate recognition for the security domain.

This is a LabeloxSec capability, gated on the active pack declaring "anpr". It is the deliberate opposite of
the AV compliance rule (services/autolabel/ocr/reader.py: OCR never reads plate text; plates are DPDPA PII
blurred by the anonymization gate). ANPR reads plates for an authorised security purpose (gate access,
perimeter, watchlist matching) and is refused entirely under a pack that does not authorise it, so it can
never run in the AV / DPDPA context.
"""
