"""HD map export: write the fused map to Lanelet2 (OSM-XML, primary) and OpenDRIVE (XML), and seal a
versioned map_commit.

Both formats now carry topology, which is what makes them maps rather than drawings. A Lanelet2 file is
defined by `relation type=lanelet` naming a left and a right boundary; this exporter used to emit the
boundaries as loose ways with no relations, so no Lanelet2 consumer could route on the result. OpenDRIVE
roads were emitted with a single centre lane of `type="none"` and no driving lanes at all, which is a
road with nothing to drive on. The pairing and chaining come from `services/hdmap/topology.py`.

A boundary that could not be paired into a lane is still exported, as a plain way in Lanelet2 and as a
road with no driving lane in OpenDRIVE, and the count is reported. Dropping it would hide real surveyed
geometry; promoting it to a lane would invent one.

Both files are stored to the object store; provenance stays on every map_element.
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


def _lane_boundaries(fused: list[dict]) -> list[dict]:
    """The lane elements, as topology input in travel order."""
    out = []
    for f in fused:
        if f.get("kind") != "lane":
            continue
        shape, coords = parse_wkt(f["wkt"])
        if shape != "line" or len(coords) < 2:
            continue
        out.append({"id": f.get("element_id") or f["wkt"][:24], "points": coords,
                    "lane_type": (f.get("attrs") or {}).get("lane_type"),
                    "confidence": float(f.get("confidence") or 0.0)})
    return out


def to_lanelet2_osm(fused: list[dict]) -> str:
    """Lanelet2 OSM-XML: boundary ways, lanelet relations over them, and signs as tagged nodes.

    The relations are the point. Lanelet2 derives routing from lanelets that share boundary ways, so
    consecutive lanelets are emitted referencing the same way ids rather than duplicate geometry, and the
    chaining computed in `topology` is written as an explicit tag as well for readers that want it
    without recomputing.
    """
    from services.hdmap.topology import build_lanelets

    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6" generator="labeloxav">']
    nid = [-1]
    wid = [-1000]
    rid = [-100000]

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

    def way(coords: list[tuple[float, float]], tags: dict) -> int:
        refs = [node(lo, la) for lo, la in coords]
        wid[0] -= 1
        lines.append(f'  <way id="{wid[0]}" visible="true">')
        lines.extend(f'    <nd ref="{r}"/>' for r in refs)
        for k, v in tags.items():
            lines.append(f'    <tag k="{escape(str(k))}" v="{escape(str(v))}"/>')
        lines.append("  </way>")
        return wid[0]

    topo = build_lanelets(_lane_boundaries(fused))
    # One way per distinct boundary polyline, shared by every lanelet that uses it, so a boundary between
    # two lanes is one way referenced twice rather than two coincident ways.
    ways: dict[tuple, int] = {}

    def boundary_way(coords: tuple, subtype: str) -> int:
        key = tuple(round(c, 9) for pt in coords for c in pt)
        if key not in ways:
            ways[key] = way(list(coords), {"type": "line_thin", "subtype": subtype or "solid"})
        return ways[key]

    lanelet_ids = []
    for ll in topo["lanelets"]:
        lw = boundary_way(ll.left, ll.left_type)
        rw = boundary_way(ll.right, ll.right_type)
        rid[0] -= 1
        lanelet_ids.append(rid[0])
        lines.append(f'  <relation id="{rid[0]}" visible="true">')
        lines.append(f'    <member type="way" ref="{lw}" role="left"/>')
        lines.append(f'    <member type="way" ref="{rw}" role="right"/>')
        lines.append('    <tag k="type" v="lanelet"/>')
        lines.append('    <tag k="subtype" v="road"/>')
        lines.append('    <tag k="location" v="urban"/>')
        lines.append('    <tag k="one_way" v="yes"/>')
        lines.append(f'    <tag k="width" v="{ll.width_m:.2f}"/>')
        lines.append(f'    <tag k="confidence" v="{ll.confidence:.2f}"/>')
        lines.append("  </relation>")

    for a, b in topo["successors"]:
        if a < len(lanelet_ids) and b < len(lanelet_ids):
            rid[0] -= 1
            lines.append(f'  <relation id="{rid[0]}" visible="true">')
            lines.append(f'    <member type="relation" ref="{lanelet_ids[a]}" role="from"/>')
            lines.append(f'    <member type="relation" ref="{lanelet_ids[b]}" role="to"/>')
            lines.append('    <tag k="type" v="lane_succession"/>')
            lines.append("  </relation>")

    # Boundaries no lane claimed are still surveyed geometry. Emitted as ways, tagged with why.
    paired: set[str] = set()
    for ll in topo["lanelets"]:
        paired.update(ll.source_elements)
    for b in _lane_boundaries(fused):
        if str(b["id"]) in paired:
            continue
        why = next((u["reason"] for u in topo["unpaired"] if str(u.get("id")) == str(b["id"])), "unpaired")
        way(b["points"], {"type": "line_thin", "subtype": b["lane_type"] or "solid",
                          "lanelet": "none", "unpaired_reason": why})

    for f in fused:
        shape, coords = parse_wkt(f["wkt"])
        if f["kind"] == "sign" and shape == "point":
            lo, la = coords[0]
            node(lo, la, {"type": "traffic_sign", "subtype": f["attrs"].get("sign_type", "sign"),
                          "category": f["attrs"].get("sign_category", ""),
                          "confidence": f"{f['confidence']:.2f}"})
    lines.append("</osm>")
    return "\n".join(lines)


def to_opendrive(fused: list[dict], center: tuple[float, float]) -> str:
    """OpenDRIVE: one road per lanelet, with a driving lane and links to the road ahead.

    Previously every road carried `<center><lane id="0" type="none"/></center>` and nothing else, so the
    file described centrelines with no lane to drive in and no way to get from one road to the next. A
    lanelet supplies both: its centreline is the road's reference line, its measured width is the lane
    width, and the successor chain becomes road linkage.

    Geometry stays piecewise `<line/>`. Fitting arcs or spirals to a traced boundary would be a second,
    lossy model of geometry already measured per point, and OpenDRIVE consumers accept polylines.
    """
    from services.hdmap.topology import build_lanelets

    clat, clon = center
    mlon = 111320.0 * math.cos(math.radians(clat))

    def xy(lon: float, lat: float) -> tuple[float, float]:
        return (lon - clon) * mlon, (lat - clat) * 111320.0

    out = ['<?xml version="1.0" encoding="UTF-8"?>', "<OpenDRIVE>",
           '  <header revMajor="1" revMinor="6" name="labeloxav" version="1.0">',
           f'    <geoReference><![CDATA[+proj=tmerc +lat_0={clat:.6f} +lon_0={clon:.6f} +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m]]></geoReference>',
           "  </header>"]

    topo = build_lanelets(_lane_boundaries(fused))
    # Road ids are one-based and follow lanelet order, so the successor pairs index straight onto them.
    succ: dict[int, int] = {}
    pred: dict[int, int] = {}
    for a, b in topo["successors"]:
        succ.setdefault(a + 1, b + 1)
        pred.setdefault(b + 1, a + 1)

    def plan_view(pts: list[tuple[float, float]]) -> tuple[list[str], float]:
        body, s0 = ["    <planView>"], 0.0
        for i in range(len(pts) - 1):
            (x0, y0), (x1, y1) = pts[i], pts[i + 1]
            seg = math.dist(pts[i], pts[i + 1])
            body.append(f'      <geometry s="{s0:.3f}" x="{x0:.3f}" y="{y0:.3f}" '
                        f'hdg="{math.atan2(y1 - y0, x1 - x0):.5f}" length="{seg:.3f}"><line/></geometry>')
            s0 += seg
        body.append("    </planView>")
        return body, s0

    for idx, ll in enumerate(topo["lanelets"], start=1):
        pts = [xy(lo, la) for lo, la in ll.centre]
        body, length = plan_view(pts)
        out.append(f'  <road name="lane{idx}" length="{length:.3f}" id="{idx}" junction="-1">')
        if idx in pred or idx in succ:
            out.append("    <link>")
            if idx in pred:
                out.append(f'      <predecessor elementType="road" elementId="{pred[idx]}" contactPoint="end"/>')
            if idx in succ:
                out.append(f'      <successor elementType="road" elementId="{succ[idx]}" contactPoint="start"/>')
            out.append("    </link>")
        out += body
        # One driving lane, centred on the reference line, with the width the pairing measured. `a` is the
        # constant term of the width polynomial, which is how OpenDRIVE states a constant width.
        out.append('    <lanes>')
        out.append('      <laneSection s="0.0">')
        out.append('        <left><lane id="1" type="driving" level="false">')
        out.append(f'          <width sOffset="0.0" a="{ll.width_m / 2.0:.3f}" b="0.0" c="0.0" d="0.0"/>')
        out.append(f'          <roadMark sOffset="0.0" type="{_road_mark(ll.left_type)}" weight="standard" color="white"/>')
        out.append("        </lane></left>")
        out.append('        <center><lane id="0" type="none" level="false"/></center>')
        out.append('        <right><lane id="-1" type="driving" level="false">')
        out.append(f'          <width sOffset="0.0" a="{ll.width_m / 2.0:.3f}" b="0.0" c="0.0" d="0.0"/>')
        out.append(f'          <roadMark sOffset="0.0" type="{_road_mark(ll.right_type)}" weight="standard" color="white"/>')
        out.append("        </lane></right>")
        out.append("      </laneSection>")
        out.append("    </lanes>")
        out.append("  </road>")

    # An unpaired boundary has no width and therefore no lane, but it is measured geometry and is kept as
    # a reference line so the file does not silently lose it.
    rid = len(topo["lanelets"])
    paired: set[str] = set()
    for ll in topo["lanelets"]:
        paired.update(ll.source_elements)
    for b in _lane_boundaries(fused):
        if str(b["id"]) in paired:
            continue
        pts = [xy(lo, la) for lo, la in b["points"]]
        if len(pts) < 2:
            continue
        body, length = plan_view(pts)
        rid += 1
        out.append(f'  <road name="boundary{rid}" length="{length:.3f}" id="{rid}" junction="-1">')
        out += body
        out.append('    <lanes><laneSection s="0.0"><center>'
                   '<lane id="0" type="none" level="false"/></center></laneSection></lanes>')
        out.append("  </road>")

    out.append("</OpenDRIVE>")
    return "\n".join(out)


def _road_mark(lane_type: str | None) -> str:
    """The ontology's lane marking type as an OpenDRIVE roadMark type."""
    return {"solid": "solid", "dashed": "broken", "double": "solid solid",
            "road_edge": "curb", "implicit": "none", "fallback": "none",
            "unknown": "none", None: "none"}.get(lane_type, "none")


def seal_map_commit_id(session_ids: list[str], fused: list[dict], calib_version: str) -> str:
    h = hashlib.sha256()
    for s in sorted(session_ids):
        h.update(s.encode())
    h.update(calib_version.encode())
    for f in sorted(fused, key=lambda x: x["wkt"]):
        h.update(f["wkt"].encode())
    return f"map-{h.hexdigest()[:16]}"
