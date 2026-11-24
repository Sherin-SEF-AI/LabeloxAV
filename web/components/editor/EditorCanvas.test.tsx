import { act, render } from "@testing-library/react";
import { forwardRef, useImperativeHandle, useRef, type ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { EdObject } from "./useEditor";

// The canvas draws through Konva, which needs a real 2D context that jsdom does not have. So Konva's
// components are replaced with plain elements that record what they were handed. That cannot say whether a
// pixel is right. It can say the things this file is responsible for: which layer a shape is in and in what
// order, which layers re-render during a gesture, and what a gesture hands back to the page.

type Rec = Record<string, unknown>;
type MockProps = { children?: ReactNode; name?: string; listening?: boolean; sceneFunc?: (ctx: unknown) => void };
const K = vi.hoisted(() => ({
  pointer: { x: 0, y: 0 },
  stage: null as Record<string, unknown> | null,
  stageHandle: null as Record<string, unknown> | null,
  layerRenders: [] as number[],
  layerSeq: 0,
  trNodes: [] as unknown[][],
  batchDraws: 0,
}));

vi.mock("@/lib/colors", () => ({ classColor: () => "#888", classFill: () => "rgba(0,0,0,0.1)" }));

vi.mock("react-konva", () => {
  const shape = (kind: string) => {
    const C = forwardRef<unknown, MockProps>(function Mock(props, ref) {
      useImperativeHandle(ref, () => ({ kind, name: props.name }), [props.name]);
      const fn = props.sceneFunc;
      return (
        <div data-k={kind} data-name={props.name}
          data-listening={props.listening === undefined ? undefined : String(props.listening)}
          ref={(el) => { if (el && fn) (el as unknown as Rec).sceneFunc = fn; }}>
          {props.children}
        </div>
      );
    });
    return C;
  };
  const Layer = ({ children, listening }: { children?: ReactNode; listening?: boolean }) => {
    // Layers mount in tree order, so the index a layer takes on its first render is its z position.
    const idx = useRef(-1);
    if (idx.current < 0) { idx.current = K.layerSeq++; K.layerRenders[idx.current] = 0; }
    K.layerRenders[idx.current] += 1;
    return <div data-k="Layer" data-listening={listening === undefined ? undefined : String(listening)}>{children}</div>;
  };
  const Stage = forwardRef<unknown, MockProps>(function Stage(props, ref) {
    K.stage = props as Rec;
    const handle = useRef<Record<string, unknown>>({ getRelativePointerPosition: () => ({ ...K.pointer }) });
    handle.current.getStage = () => handle.current;
    K.stageHandle = handle.current;
    useImperativeHandle(ref, () => handle.current, []);
    return <div data-k="Stage">{props.children}</div>;
  });
  const Transformer = forwardRef<unknown, Rec>(function Transformer(_props, ref) {
    useImperativeHandle(ref, () => ({
      nodes: (n: unknown[]) => { K.trNodes.push(n); },
      getLayer: () => ({ batchDraw: () => { K.batchDraws += 1; } }),
    }), []);
    return <div data-k="Transformer" />;
  });
  return {
    Stage, Layer, Transformer,
    Circle: shape("Circle"), Group: shape("Group"), Image: shape("Image"), Line: shape("Line"),
    Rect: shape("Rect"), Shape: shape("Shape"), Text: shape("Text"),
  };
});

import EditorCanvas from "./EditorCanvas";

function obj(id: string, over: Partial<EdObject> = {}): EdObject {
  return {
    id, class_id: 1, class_name: "sedan", bbox: [10, 10, 50, 40], mask: [], attrs: {},
    conf: 0.9, state: "review", visible: true, ...over,
  };
}

const handlers = () => ({
  onViewport: vi.fn(), onSelect: vi.fn(), onSelectMany: vi.fn(), onUpdateBbox: vi.fn(), onDrawBox: vi.fn(),
  onDrawAmodal: vi.fn(), onDrawPolygon: vi.fn(), onDrawPolyline: vi.fn(), onDrawAdverse: vi.fn(),
  onPlaceCuboid: vi.fn(), onMagicWand: vi.fn(), onBrushStroke: vi.fn(), onPickSuperpixel: vi.fn(),
  onPlaceKeypoint: vi.fn(), onUpdateKeypoints: vi.fn(), onSamPoint: vi.fn(), onSamBox: vi.fn(),
  onUpdateMask: vi.fn(), onCursor: vi.fn(), onImageLoad: vi.fn(), onContextMenu: vi.fn(),
});

const VIEW = { scale: 1, ox: 0, oy: 0 };
const OBJECTS = [
  obj("a", { mask: [[0, 0, 9, 0, 9, 9]] }),
  obj("line", { polyline: [[0, 0], [20, 20]] }),
  obj("pose", { bbox: [60, 60, 90, 90], keypoints: { skeleton: "person_17", points: [[70, 70, 2]] } }),
];

function mount(over: Rec = {}, h = handlers()) {
  const props = {
    imageUrl: "/api/frames/f1/image", imgW: 200, imgH: 100, objects: OBJECTS, selectedId: "a" as string | null,
    tool: "select", candidate: null, viewport: VIEW, panning: false, brushRadius: 14, ...h, ...over,
  } as unknown as Parameters<typeof EditorCanvas>[0];
  const view = render(<EditorCanvas {...props} />);
  return { view, h, props };
}

const stage = () => K.stage as Record<string, (e: unknown) => void>;
const at = (x: number, y: number) => { K.pointer = { x, y }; };
const onEmpty = (over: Rec = {}) => ({ target: K.stageHandle, evt: { shiftKey: false, metaKey: false, ...over } });
const down = (x: number, y: number, e: unknown = onEmpty()) => act(() => { at(x, y); stage().onMouseDown(e); });
const move = (x: number, y: number) => act(() => { at(x, y); stage().onMouseMove({}); });
const up = () => act(() => { stage().onMouseUp({}); });
const layers = (c: HTMLElement) => Array.from(c.querySelectorAll('[data-k="Layer"]')) as HTMLElement[];
const kinds = (el: Element, sel = "[data-k]") => Array.from(el.querySelectorAll(sel)).map((n) => n.getAttribute("data-k"));

// A stand-in for the browser's Image: jsdom never fires onload. Loads on demand.
class FakeImage {
  static made: FakeImage[] = [];
  src = ""; crossOrigin: string | null = null; onload: (() => void) | null = null; onerror: (() => void) | null = null;
  constructor() { FakeImage.made.push(this); }
}
const RealImage = window.Image;

beforeEach(() => {
  K.layerRenders = []; K.layerSeq = 0; K.trNodes = []; K.batchDraws = 0; K.stage = null; K.pointer = { x: 0, y: 0 };
  FakeImage.made = [];
  window.Image = FakeImage as unknown as typeof window.Image;
});
afterEach(() => { window.Image = RealImage; });

describe("three layers", () => {
  it("are a static layer that is never hit-tested, the objects, and the interaction on top", () => {
    const { view } = mount();
    const ls = layers(view.container);
    expect(ls).toHaveLength(3);
    expect(ls[0].getAttribute("data-listening")).toBe("false");
    expect(ls[1].getAttribute("data-listening")).toBeNull();
    expect(ls[2].getAttribute("data-listening")).toBeNull();
  });

  it("put the frame image in the static layer and nowhere else", () => {
    const { view } = mount();
    act(() => { FakeImage.made[0].onload?.(); });
    const ls = layers(view.container);
    expect(kinds(ls[0])).toEqual(["Image"]);
    expect(kinds(ls[1])).not.toContain("Image");
    expect(kinds(ls[2])).not.toContain("Image");
  });

  it("put every object shape in the objects layer, in the order the single layer had", () => {
    // Order is hit-testing: polylines under boxes, boxes under the selected mask's handles, keypoints last.
    const { view } = mount();
    const named = Array.from(layers(view.container)[1].children).map(
      (n) => `${n.getAttribute("data-k")}${n.getAttribute("data-name") ? `:${n.getAttribute("data-name")}` : ""}`);
    const first = (s: string) => named.indexOf(s);
    expect(first("Shape")).toBe(0);                                  // the mask
    expect(first("Line:obj:line")).toBeGreaterThan(first("Shape"));  // then the polyline
    expect(first("Rect:obj:pose")).toBeGreaterThan(first("Line:obj:line"));
    // The selected box is the last box, so its handles are never under a neighbour.
    expect(named.filter((s) => s.startsWith("Rect:obj:")).pop()).toBe("Rect:obj:a");
    expect(named).not.toContain("Rect:obj:line");                    // a polyline object draws no box
    expect(named.lastIndexOf("Group")).toBe(named.length - 1);       // committed keypoints on top
    expect(named.indexOf("Circle")).toBeGreaterThan(first("Rect:obj:a"));  // mask vertex handles above boxes
  });

  it("put the transform handles in the interaction layer and attach them to the selected box", () => {
    const { view } = mount();
    const ls = layers(view.container);
    expect(kinds(ls[2])).toContain("Transformer");
    expect(kinds(ls[1])).not.toContain("Transformer");
    expect(K.trNodes.at(-1)).toEqual([{ kind: "Rect", name: "obj:a" }]);
    expect(K.batchDraws).toBeGreaterThan(0);
  });

  it("detach the handles when nothing is selected, and drop them outside the select tool", () => {
    const { view } = mount({ selectedId: null });
    expect(K.trNodes.at(-1)).toEqual([]);
    view.unmount();
    K.layerSeq = 0;
    const brush = mount({ tool: "brush" });
    expect(kinds(layers(brush.view.container)[2])).not.toContain("Transformer");
  });

  it("honour the layer toggles", () => {
    const off = { boxes: false, masks: false, labels: false, lanes: false, drivable: false, adverse: false, cuboids: false, seg: false };
    const { view } = mount({ layers: off, drivable: { drivable: [[0, 0, 5, 0, 5, 5]] },
                             adverse: [{ region_id: "r", geometry: [0, 0, 5, 0, 5, 5], condition: "glare" }] });
    const ls = layers(view.container);
    expect(kinds(ls[0])).toEqual([]);
    expect(kinds(ls[1]).filter((k) => k === "Rect" || k === "Shape")).toEqual([]);
  });

  it("draw drivable and adverse regions in the static layer", () => {
    const { view } = mount({ drivable: { drivable: [[0, 0, 5, 0, 5, 5]], fallback: [[1, 1, 2, 1, 2, 2]] },
                             adverse: [{ region_id: "r", geometry: [0, 0, 5, 0, 5, 5], condition: "glare" }] });
    expect(kinds(layers(view.container)[0])).toEqual(["Line", "Line", "Line"]);
  });
});

describe("a gesture redraws the interaction layer and nothing under it", () => {
  it("a brush stroke", () => {
    const { h } = mount({ tool: "brush", selectedId: null });
    const before = [...K.layerRenders];
    down(10, 10); move(14, 11); move(30, 30); move(31, 30);
    expect(K.layerRenders[0]).toBe(before[0]);
    expect(K.layerRenders[1]).toBe(before[1]);
    expect(K.layerRenders[2]).toBeGreaterThan(before[2]);
    up();
    expect(h.onBrushStroke).toHaveBeenCalledTimes(1);
  });

  it("a marquee", () => {
    mount({ selectedId: null });
    const before = [...K.layerRenders];
    down(0, 0); move(40, 40); move(100, 100);
    expect(K.layerRenders.slice(0, 2)).toEqual(before.slice(0, 2));
    expect(K.layerRenders[2]).toBeGreaterThan(before[2]);
    up();
  });

  it("a box being drawn", () => {
    mount({ tool: "box", selectedId: null });
    const before = [...K.layerRenders];
    down(5, 5); move(25, 25);
    expect(K.layerRenders.slice(0, 2)).toEqual(before.slice(0, 2));
    up();
  });

  it("moving the pointer with no gesture in progress redraws nothing at all", () => {
    const { h } = mount();
    const before = [...K.layerRenders];
    move(40, 40); move(41, 41);
    expect(K.layerRenders).toEqual(before);
    expect(h.onCursor).toHaveBeenLastCalledWith([41, 41]);
  });

  it("the page re-rendering with the same props redraws nothing either", () => {
    const { view, props } = mount();
    const before = [...K.layerRenders];
    view.rerender(<EditorCanvas {...props} />);
    view.rerender(<EditorCanvas {...props} />);
    expect(K.layerRenders).toEqual(before);
  });

  it("a new selection redraws the objects and leaves the image alone", () => {
    const { view, props } = mount();
    const before = [...K.layerRenders];
    view.rerender(<EditorCanvas {...props} selectedId="pose" />);
    expect(K.layerRenders[0]).toBe(before[0]);
    expect(K.layerRenders[1]).toBeGreaterThan(before[1]);
  });
});

describe("the brush stroke", () => {
  it("previews as one shape however long the stroke is, not a node per stamp", () => {
    const { view } = mount({ tool: "brush", selectedId: null, objects: [] });
    down(10, 10);
    for (let i = 1; i <= 40; i++) move(10 + i, 10);
    const top = layers(view.container)[2];
    expect(kinds(top)).toEqual(["Shape"]);
    up();
  });

  it("traces exactly the stamps it will commit", () => {
    const { view, h } = mount({ tool: "eraser", selectedId: null, objects: [], brushRadius: 6 });
    down(10, 10); move(20, 10); move(30, 12);
    const shape = layers(view.container)[2].querySelector('[data-k="Shape"]') as unknown as Rec;
    const arcs: number[][] = [];
    const ctx = { beginPath() {}, moveTo() {}, fill() {}, fillStyle: "",
                  arc: (x: number, y: number, r: number) => { arcs.push([x, y, r]); } };
    (shape.sceneFunc as (c: unknown) => void)(ctx);
    expect(ctx.fillStyle).toBe("rgba(248,81,73,0.25)");
    up();
    const ops = h.onBrushStroke.mock.calls[0][0] as { op: string; center: number[]; radius: number }[];
    expect(ops).toEqual([
      { op: "erase", center: [10, 10], radius: 6 },
      { op: "erase", center: [20, 10], radius: 6 },
      { op: "erase", center: [30, 12], radius: 6 },
    ]);
    expect(arcs).toEqual(ops.map((o) => [...o.center, o.radius]));
  });

  it("clears the preview once committed", () => {
    const { view } = mount({ tool: "brush", selectedId: null, objects: [] });
    down(10, 10); move(20, 10); up();
    expect(kinds(layers(view.container)[2])).toEqual([]);
  });
});

describe("what a gesture hands the page", () => {
  it("a drawn box, normalised whichever way it was dragged", () => {
    const { h } = mount({ tool: "box", selectedId: null });
    down(30, 40); move(10, 5); up();
    expect(h.onDrawBox).toHaveBeenCalledWith([10, 5, 30, 40]);
  });

  it("ignores a box too small to be one", () => {
    const { h } = mount({ tool: "box", selectedId: null });
    down(30, 40); move(31, 41); up();
    expect(h.onDrawBox).not.toHaveBeenCalled();
  });

  it("a marquee selects what it encloses, skipping locked and hidden objects", () => {
    const objects = [obj("in"), obj("locked", { locked: true }), obj("hidden", { visible: false }), obj("out", { bbox: [150, 10, 190, 40] })];
    const { h } = mount({ objects, selectedId: null });
    down(0, 0, onEmpty({ shiftKey: true })); move(100, 100); up();
    expect(h.onSelectMany).toHaveBeenCalledWith(["in"], true);
  });

  it("a click on empty canvas clears the selection rather than starting a marquee", () => {
    const { h } = mount();
    down(120, 80); up();
    expect(h.onSelect).toHaveBeenCalledWith(null);
    expect(h.onSelectMany).not.toHaveBeenCalled();
  });

  it("nothing while space-panning", () => {
    const { h } = mount({ tool: "box", panning: true, selectedId: null });
    down(5, 5); move(50, 50); up();
    expect(h.onDrawBox).not.toHaveBeenCalled();
  });
});

describe("the frame image", () => {
  it("tells the page when it has loaded, which is when the neighbours start warming", () => {
    const { h } = mount();
    expect(FakeImage.made[0].src).toBe("/api/frames/f1/image");
    expect(FakeImage.made[0].crossOrigin).toBe("anonymous");
    expect(h.onImageLoad).not.toHaveBeenCalled();
    act(() => { FakeImage.made[0].onload?.(); });
    expect(h.onImageLoad).toHaveBeenCalledWith("/api/frames/f1/image");
  });

  it("ignores an image that arrives after the frame has changed", () => {
    // Otherwise the previous frame's pixels land under the next frame's boxes.
    const { view, props, h } = mount();
    view.rerender(<EditorCanvas {...props} imageUrl="/api/frames/f2/image" />);
    act(() => { FakeImage.made[0].onload?.(); });
    expect(h.onImageLoad).not.toHaveBeenCalled();
    act(() => { FakeImage.made[1].onload?.(); });
    expect(h.onImageLoad).toHaveBeenCalledWith("/api/frames/f2/image");
  });

  it("does not load the image again because the page handed it a new callback", () => {
    const { view, props } = mount();
    view.rerender(<EditorCanvas {...props} onImageLoad={vi.fn()} />);
    expect(FakeImage.made).toHaveLength(1);
  });

  it("says nothing after the canvas is gone", () => {
    const { view, h } = mount();
    view.unmount();
    FakeImage.made[0].onload?.();
    FakeImage.made[0].onerror?.();
    expect(h.onImageLoad).not.toHaveBeenCalled();
  });
});

describe("fit", () => {
  it("asks for a fit when the viewport is pending and the container has a size", () => {
    const sized = vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockReturnValue(400);
    const tall = vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockReturnValue(400);
    const { h } = mount({ viewport: { scale: 0, ox: 0, oy: 0 } });
    sized.mockRestore(); tall.mockRestore();
    // 200x100 into 400x400 at 96%: scale 1.92, centred.
    expect(h.onViewport).toHaveBeenCalledWith({ scale: 1.92, ox: 8, oy: 104 });
  });

  it("does not refit a viewport that is already set", () => {
    const { h } = mount();
    expect(h.onViewport).not.toHaveBeenCalled();
  });
});
