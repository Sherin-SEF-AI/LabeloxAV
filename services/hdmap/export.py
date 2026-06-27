"""HD map export (M3.3): write the fused map to Lanelet2 (OSM-XML, primary) and OpenDRIVE (XML), and seal
a versioned map_commit. Lanelet2 emits lane boundaries as tagged ways and signs as tagged nodes;
OpenDRIVE emits each lane as a road with a polyline planView in a local transverse-mercator metric frame
(geoReference in the header). Both are stored to the object store; provenance stays on every map_element.
"""

from __future__ import annotations

import hashlib
import math
from xml.sax.saxutils import escape


def parse_wkt(wkt: str) -> tuple[str, list[tuple[float, float]]]:
    kind = "line" if wkt.upper().startswith("LINESTRING") else "point"
    body = wkt[wkt.index("(") + 1: wkt.rindex(")")]
    coords = [(float(p.split()[0]), float(p.split()[1])) for p in body.split(",")]
    return kind, coords


def to_lanelet2_osm(fused: list[dict]) -> str:
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6" generator="labeloxav">']
    nid = [-1]
    wid = [-1000]

    def node(lon: float, lat: float, tags: dict | None = None) -> int:
        nid[0] -= 1
        if tags:
            lines.append(f'  <node id="{nid[0]}" lat="{lat:.8f}" lon="{lon:.8f}" visible="true">')
            for k, v in tags.items():
                lines.append(f'    <tag k="{escape(str(k))}" v="{escape(str(v))}"/>')
            lines.append("  </node>")
        else:
            lines.append(f'  <node id="{nid[0]}" lat="{lat:.8f}" lon="{lon:.8f}" visible="true"/>')
        return nid[0]

    for f in fused:
        kind, coords = parse_wkt(f["wkt"])
        if f["kind"] == "lane" and kind == "line" and len(coords) >= 2:
            refs = [node(lo, la) for lo, la in coords]
            wid[0] -= 1
            lines.append(f'  <way id="{wid[0]}" visible="true">')
            lines += [f'    <nd ref="{r}"/>' for r in refs]
            lines.append('    <tag k="type" v="line_thin"/>')
            lines.append(f'    <tag k="subtype" v="{escape(str(f["attrs"].get("lane_type", "solid")))}"/>')
            lines.append(f'    <tag k="confidence" v="{f["confidence"]:.2f}"/>')
            lines.append("  </way>")
        elif f["kind"] == "sign" and kind == "point":
            lo, la = coords[0]
            node(lo, la, {"type": "traffic_sign", "subtype": f["attrs"].get("sign_type", "sign"),
                          "category": f["attrs"].get("sign_category", ""), "confidence": f"{f['confidence']:.2f}"})
    lines.append("</osm>")
    return "\n".join(lines)


def to_opendrive(fused: list[dict], center: tuple[float, float]) -> str:
    clat, clon = center
    mlon = 111320.0 * math.cos(math.radians(clat))

    def xy(lon: float, lat: float) -> tuple[float, float]:
        return (lon - clon) * mlon, (lat - clat) * 111320.0

    out = ['<?xml version="1.0" encoding="UTF-8"?>', "<OpenDRIVE>",
           '  <header revMajor="1" revMinor="6" name="labeloxav" version="1.0">',
           f'    <geoReference><![CDATA[+proj=tmerc +lat_0={clat:.6f} +lon_0={clon:.6f} +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m]]></geoReference>',
           "  </header>"]
    rid = 0
    for f in fused:
        if f["kind"] != "lane":
            continue
        kind, coords = parse_wkt(f["wkt"])
        if kind != "line" or len(coords) < 2:
            continue
        pts = [xy(lo, la) for lo, la in coords]
        length = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
        rid += 1
        out.append(f'  <road name="lane{rid}" length="{length:.3f}" id="{rid}" junction="-1">')
        out.append("    <planView>")
        s = 0.0
        for i in range(len(pts) - 1):
            (x0, y0), (x1, y1) = pts[i], pts[i + 1]
            seg = math.dist(pts[i], pts[i + 1])
            out.append(f'      <geometry s="{s:.3f}" x="{x0:.3f}" y="{y0:.3f}" hdg="{math.atan2(y1 - y0, x1 - x0):.5f}" length="{seg:.3f}"><line/></geometry>')
            s += seg
        out.append('    </planView>')
        out.append('    <lanes><laneSection s="0.0"><center><lane id="0" type="none" level="false"/></center></laneSection></lanes>')
        out.append("  </road>")
    out.append("</OpenDRIVE>")
    return "\n".join(out)


def seal_map_commit_id(session_ids: list[str], fused: list[dict], calib_version: str) -> str:
    h = hashlib.sha256()
    for s in sorted(session_ids):
        h.update(s.encode())
    h.update(calib_version.encode())
    for f in sorted(fused, key=lambda x: x["wkt"]):
        h.update(f["wkt"].encode())
    return f"map-{h.hexdigest()[:16]}"
