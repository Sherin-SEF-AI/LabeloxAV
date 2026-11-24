import { act, render } from "@testing-library/react";
import { forwardRef, useImperativeHandle, useRef, type ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";

import type { EdObject } from "./useEditor";

// The canvas on a view-only frame. Konva is replaced, as in EditorCanvas.test.tsx, by elements that
// record what they were handed; here what matters is which of them can be dragged and which handlers
// still exist, because "view only" on a canvas means precisely that nothing can be picked up.

type Rec = Record<string, unknown>;
// Named props only. An index signature here would be flattened by forwardRef's own prop mapping and
// `children` would come out as unknown.
type MockProps = {
  children?: ReactNode; draggable?: boolean; name?: string;
  onDragEnd?: unknown; onMouseDown?: unknown; onWheel?: unknown;
};
const K = vi.hoisted(() => ({
  shapes: [] as Rec[],
  stage: null as Rec | null,
  stageHandle: null as Rec | null,
  pointer: { x: 0, y: 0 },
}));

vi.mock("@/lib/colors", () => ({ classColor: () => "#888", classFill: () => "rgba(0,0,0,0.1)" }));

vi.mock("react-konva", () => {
  const shape = (kind: string) => forwardRef<unknown, MockProps>(function Mock(props, ref) {
    useImperativeHandle(ref, () => ({ kind }), []);
    K.shapes.push({ kind, ...props });
    return <div data-k={kind} data-draggable={props.draggable ? "true" : "false"}>{props.children}</div>;
  });
  const Layer = ({ children }: { children?: ReactNode }) => <div data-k="Layer">{children}</div>;
  const Stage = forwardRef<unknown, MockProps>(function Stage(props, ref) {
    K.stage = props as Rec;
    const handle = useRef<Rec>({
      getRelativePointerPosition: () => ({ ...K.pointer }), getPointerPosition: () => ({ ...K.pointer }),
    });
    handle.current.getStage = () => handle.current;
    K.stageHandle = handle.current;
    useImperativeHandle(ref, () => handle.current, []);
    return <div data-k="Stage">{props.children}</div>;
  });
  const Transformer = forwardRef<unknown, Rec>(function Transformer(_props, ref) {
    useImperativeHandle(ref, () => ({ nodes: () => {}, getLayer: () => ({ batchDraw: () => {} }) }), []);
    return <div data-k="Transformer" />;
  });
  return {
    Stage, Layer, Transformer,
    Circle: shape("Circle"), Group: shape("Group"), Image: shape("Image"), Line: shape("Line"),
    Rect: shape("Rect"), Shape: shape("Shape"), Text: shape("Text"),
  };
});

import EditorCanvas from "./EditorCanvas";

const OBJECTS: EdObject[] = [
  { id: "a", class_id: 1, class_name: "sedan", bbox: [10, 10, 50, 40], mask: [[10, 10, 50, 10, 50, 40]],
    attrs: {}, conf: 0.9, state: "review", visible: true },
  { id: "pose", class_id: 2, class_name: "pedestrian", bbox: [60, 60, 90, 90], mask: [], attrs: {}, conf: 0.9,
    state: "review", visible: true, keypoints: { skeleton: "person_17", points: [[70, 70, 2], [80, 80, 2]] } },
];

const handlers = () => ({
  onViewport: vi.fn(), onSelect: vi.fn(), onSelectMany: vi.fn(), onUpdateBbox: vi.fn(), onDrawBox: vi.fn(),
  onDrawAmodal: vi.fn(), onDrawPolygon: vi.fn(), onDrawPolyline: vi.fn(), onDrawAdverse: vi.fn(),
  onPlaceCuboid: vi.fn(), onMagicWand: vi.fn(), onBrushStroke: vi.fn(), onPickSuperpixel: vi.fn(),
  onPlaceKeypoint: vi.fn(), onUpdateKeypoints: vi.fn(), onSamPoint: vi.fn(), onSamBox: vi.fn(),
  onUpdateMask: vi.fn(), onCursor: vi.fn(),
});

function mount(over: Rec = {}) {
  K.shapes = [];
  const h = handlers();
  const props = {
    imageUrl: "/api/frames/f1/image", imgW: 200, imgH: 100, objects: OBJECTS, selectedId: "a",
    tool: "select", candidate: null, viewport: { scale: 1, ox: 0, oy: 0 }, panning: false, ...h, ...over,
  } as unknown as Parameters<typeof EditorCanvas>[0];
  const view = render(<EditorCanvas {...props} />);
  return { view, h };
}

const draggable = (c: HTMLElement) => c.querySelectorAll('[data-draggable="true"]').length;
const transformers = (c: HTMLElement) => c.querySelectorAll('[data-k="Transformer"]').length;
const boxes = () => K.shapes.filter((s) => s.kind === "Rect" && typeof s.name === "string" && s.onDragEnd);

describe("an editable canvas, for contrast", () => {
  it("lets the selected box be dragged and resized, and its mask reshaped", () => {
    const { view } = mount();
    // The selected box, and a handle per mask vertex.
    expect(draggable(view.container)).toBe(1 + 3);
    expect(transformers(view.container)).toBe(1);
  });

  it("lets a selected pose's points be dragged", () => {
    const { view } = mount({ selectedId: "pose" });
    expect(draggable(view.container)).toBe(1 + 2);
  });
});

describe("a view-only canvas", () => {
  it("has nothing that can be dragged and no transform handles", () => {
    for (const selectedId of ["a", "pose"]) {
      const { view } = mount({ readOnly: true, selectedId });
      expect(draggable(view.container), selectedId).toBe(0);
      expect(transformers(view.container), selectedId).toBe(0);
      view.unmount();
    }
  });

  it("draws no mask vertex or midpoint handles, so a mask cannot be reshaped or a vertex deleted", () => {
    const editable = mount();
    const withHandles = editable.view.container.querySelectorAll('[data-k="Circle"]').length;
    editable.view.unmount();
    const { view } = mount({ readOnly: true });
    const without = view.container.querySelectorAll('[data-k="Circle"]').length;
    // Three vertices and three midpoints on the selected mask; the pose's two points are still drawn.
    expect(withHandles - without).toBe(6);
    expect(without).toBe(2);
  });

  it("still selects an object that is clicked", () => {
    const { h } = mount({ readOnly: true, selectedId: null });
    const box = boxes().find((s) => s.name === "obj:pose");
    expect(box).toBeDefined();
    act(() => { (box!.onMouseDown as (e: Rec) => void)({ cancelBubble: false }); });
    expect(h.onSelect).toHaveBeenCalledWith("pose");
  });

  it("still zooms", () => {
    const { h } = mount({ readOnly: true });
    const stage = K.stage as Record<string, (e: unknown) => void>;
    K.pointer = { x: 20, y: 20 };
    act(() => { stage.onWheel({ evt: { preventDefault: () => {}, deltaY: -1 } }); });
    expect(h.onViewport).toHaveBeenCalled();
  });

  it("gets its handles back when the lock returns", () => {
    const { view } = mount({ readOnly: true });
    expect(transformers(view.container)).toBe(0);
    view.unmount();
    const again = mount({ readOnly: false });
    expect(transformers(again.view.container)).toBe(1);
    expect(draggable(again.view.container)).toBe(4);
  });
});
