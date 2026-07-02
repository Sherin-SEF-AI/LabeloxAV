"""A small weather-forecast source for the Fleet Dispatch agent. Keyless (open-meteo) and gated off by
default so the fleet logic works offline and in tests; enable with LBX_WEATHER_ENABLED=1. Any failure
degrades to 'unknown' so a forecast is never a hard dependency."""

from __future__ import annotations

import os

from core.logging import get_logger

log = get_logger("agent.weather")

# rough city centroids for the corpus's Indian cities (3-letter code prefix -> lat, lon)
_CITY_LATLON = {
    "BLR": (12.97, 77.59), "DEL": (28.61, 77.21), "BOM": (19.08, 72.88), "MAA": (13.08, 80.27),
    "HYD": (17.39, 78.49), "CCU": (22.57, 88.36), "PNQ": (18.52, 73.86), "AMD": (23.02, 72.57),
}


def city_latlon(city: str | None) -> tuple[float, float] | None:
    return _CITY_LATLON.get((city or "").upper()[:3])


def _condition(weathercode: int, precip_prob: float) -> str:
    if weathercode in (45, 48):
        return "fog"
    if weathercode >= 51 or precip_prob >= 50:
        return "rain"
    return "clear"


async def forecast(city: str | None) -> dict:
    """Next-window forecast condition for a city: clear | rain | fog | unknown. Network-gated + graceful."""
    if not os.environ.get("LBX_WEATHER_ENABLED"):
        return {"condition": "unknown", "source": "disabled"}
    ll = city_latlon(city)
    if ll is None:
        return {"condition": "unknown", "source": "no_latlon"}
    try:
        import httpx

        resp = httpx.get("https://api.open-meteo.com/v1/forecast",
                         params={"latitude": ll[0], "longitude": ll[1],
                                 "hourly": "precipitation_probability,weathercode", "forecast_days": 1},
                         timeout=5)
        resp.raise_for_status()
        h = resp.json().get("hourly", {})
        codes = h.get("weathercode", [0]) or [0]
        probs = h.get("precipitation_probability", [0]) or [0]
        # look at the evening window (indices 18..22) if present, else the max of the day
        idx = range(18, 23) if len(codes) > 22 else range(len(codes))
        code = max((codes[i] for i in idx), default=0)
        prob = max((probs[i] for i in idx), default=0)
        return {"condition": _condition(int(code), float(prob)), "source": "open-meteo", "precip_prob": prob}
    except Exception as exc:  # noqa: BLE001
        log.warning("weather.failed", city=city, error=str(exc))
        return {"condition": "unknown", "source": "error"}
