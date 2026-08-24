import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { FrameMeta, Ontology } from "@/lib/types";
import PropertiesPanel, { type PropertiesPanelProps } from "./PropertiesPanel";
import type { EdObject, EditorState } from "../useEditor";

// The integration test for a panel that came out of a 2,318-line page with no coverage. Two of these
// assertions are about the failure modes the extraction itself introduces.
//
// Keep-alive is the important one. Agent, bulk edit and the frame-data sections all shared a single tab
// before, so switching between them never unmounted anything. Splitting them across three tabs makes a
// naive conditional render discard an in-flight dry-run plan, a typed prompt and the reanalysis findings
// on every switch. Nothing throws when that happens; the work is just gone.

vi.mock("@/lib/colors", () => ({ classColor: () => "#888" }));
vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }) }));

// The four tool panels each fetch on mount and are tested elsewhere or not at all; what matters here is
// where they are mounted and when.
//
// Counted in an effect with empty deps, not in the render body. A render body counts renders, and
// PropertiesPanel re-renders its children on every tab change whether or not they were unmounted, so that
// version of this test fails on working code and would have been "fixed" by loosening the assertion.
let agentMounts = 0;
vi.mock("@/components/agent/AgentPanel", async () => {
  const { useEffect } = await import("react");
  // Named, and capitalised, so react-hooks/rules-of-hooks recognises it as a component rather than a bare
  // function calling a hook.
  function MockAgentPanel() {
    useEffect(() => { agentMounts += 1; }, []);
    return <div data-testid="agent">agent panel</div>;
  }
  return { default: MockAgentPanel };
});
vi.mock("@/components/agent/BulkEditBar", () => ({
  default: () => <div data-testid="bulk">bulk edit bar</div>,
}));
vi.mock("@/components/editor/SceneGraphPanel", () => ({
  default: () => <div data-testid="scene">scene graph</div>,
}));
vi.mock("@/components/editor/HistoryPanel", () => ({
  default: () => <div data-testid="history">history</div>,
}));

const ONTO = {
  classes: [
    { id: 1, name: "sedan", l0: "vehicle", l1: "car", india: false },
    { id: 2, name: "autorickshaw", l0: "vehicle", l1: "three_wheeler", india: true },
  ],
  attributes: {},
  attribute_scope: {},
} as unknown as Ontology;

const META = { session_id: "s1", is_lidar: false } as unknown as FrameMeta;

function obj(id: string, over: Partial<EdObject> = {}): EdObject {
  return {
    id, class_id: 1, class_name: "sedan", bbox: [0, 0, 10, 10], mask: [], attrs: {},
    conf: 0.9, state: "review", visible: true, ...over,
  };
}

function state(objects: EdObject[], over: Partial<EditorState> = {}): EditorState {
  return {
    objects, deleted: [], selectedId: null, selectedIds: [], tool: "select",
    viewport: { scale: 1, ox: 0, oy: 0 }, candidate: null, touched: [], past: [], future: [], ...over,
  };
}

function panel(over: {
  objects?: EdObject[]; selected?: EdObject | null; meta?: FrameMeta; st?: Partial<EditorState>;
} = {}) {
  const objects = over.objects ?? [obj("aaaaaaaa11")];
  const props: PropertiesPanelProps = {
    frame: { id: "f1", meta: over.meta ?? META, onto: ONTO, dirty: false, flash: vi.fn() },
    editor: { st: state(objects, over.st), dispatch: vi.fn(), selected: over.selected ?? null },
    onCollapse: vi.fn(),
    klass: { current: ONTO.classes[0], onPick: vi.fn(), onAdd: vi.fn() },
    sel: {
      relationships: [], linkKind: "rider_of", linkFrom: null, dynamics: {},
      onSetAttr: vi.fn(), onLinkKind: vi.fn(), onToggleLink: vi.fn(),
      onDeleteRelationship: vi.fn(), onRecomputeDynamics: vi.fn(),
    },
    tools: {
      lanes: [], hasDrivable: false, onSegRoad: vi.fn(), onProposeLanes: vi.fn(),
      onAgentApplied: vi.fn(), onHistoryRestored: vi.fn(), onLiftCuboids: vi.fn(),
    },
  };
  return <PropertiesPanel {...props} />;
}

describe("objects and tools are no longer exclusive", () => {
  it("shows the object list and the tools at the same time", () => {
    // The whole point of the redesign. Before this, the object you were labelling and the agent labelling
    // it were on two tabs that could not both be open.
    render(panel());
    expect(screen.getByText("aaaaaaaa")).toBeInTheDocument();
    expect(screen.getByTestId("agent")).toBeInTheDocument();
  });
});

describe("keep-alive across tool tabs", () => {
  it("does not remount the agent when you leave and come back", async () => {
    agentMounts = 0;
    render(panel());
    expect(agentMounts).toBe(1);
    await userEvent.click(screen.getByRole("tab", { name: "bulk edit" }));
    await userEvent.click(screen.getByRole("tab", { name: "agent" }));
    // A remount here means a dry-run plan, a typed prompt and the reanalysis findings were thrown away.
    expect(agentMounts).toBe(1);
    expect(screen.getByTestId("agent")).toBeVisible();
  });

  it("mounts a tool group only once its tab is first visited", async () => {
    render(panel());
    // Not merely hidden: absent. Mounting all three up front would fire four requests on panel open for
    // tabs nobody has clicked.
    expect(screen.queryByTestId("bulk")).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("tab", { name: "bulk edit" }));
    expect(screen.getByTestId("bulk")).toBeVisible();
  });

  it("hides the inactive group rather than showing two at once", async () => {
    render(panel());
    await userEvent.click(screen.getByRole("tab", { name: "bulk edit" }));
    expect(screen.getByTestId("agent")).not.toBeVisible();
    expect(screen.getByTestId("bulk")).toBeVisible();
  });
});

describe("frame data tab", () => {
  it("carries history, scene graph and road segmentation", async () => {
    render(panel());
    await userEvent.click(screen.getByRole("tab", { name: "frame data" }));
    expect(screen.getByText("history and saves")).toBeInTheDocument();
    expect(screen.getByText("scene graph + vlm dataset")).toBeInTheDocument();
    expect(screen.getByText("road segmentation")).toBeInTheDocument();
  });

  it("offers the LiDAR lift only on a frame that carries a point cloud", async () => {
    render(panel());
    await userEvent.click(screen.getByRole("tab", { name: "frame data" }));
    expect(screen.queryByText("lidar bev")).not.toBeInTheDocument();
  });

  it("offers it when the frame does", async () => {
    render(panel({ meta: { session_id: "s1", is_lidar: true, lidar_points: 120000 } as unknown as FrameMeta }));
    await userEvent.click(screen.getByRole("tab", { name: "frame data" }));
    expect(screen.getByText("lidar bev")).toBeInTheDocument();
    expect(screen.getByText("120,000 pts")).toBeInTheDocument();
  });
});

describe("the selection band", () => {
  it("is absent with nothing selected, and the object list is still there", () => {
    render(panel());
    expect(screen.queryByText("attributes")).not.toBeInTheDocument();
    expect(screen.queryByText("dynamics")).not.toBeInTheDocument();
    expect(screen.getByText("aaaaaaaa")).toBeInTheDocument();
  });

  it("appears above the object list when something is selected", () => {
    const o = obj("aaaaaaaa11");
    render(panel({ objects: [o], selected: o }));
    expect(screen.getByText("attributes")).toBeInTheDocument();
    expect(screen.getByText("dynamics")).toBeInTheDocument();
  });
});

describe("the header", () => {
  it("names the class being painted with, without the full list taking up the panel", () => {
    render(panel());
    const card = screen.getByText("painting as").parentElement;
    // Scoped to the header card: "sedan" is also the object list's group header for the same class.
    expect(card).toHaveTextContent("sedan");
    // The old palette kept a search box and a scrolling list permanently expanded here, roughly 180px of
    // panel that an annotator working one class for an hour never touched.
    expect(screen.queryByLabelText("search or add class")).not.toBeInTheDocument();
  });

  it("opens the class list on demand", async () => {
    render(panel());
    await userEvent.click(screen.getByRole("button", { name: "change" }));
    expect(screen.getByLabelText("search or add class")).toBeInTheDocument();
  });

  it("counts review progress on the states the database admits", () => {
    render(panel({
      objects: [
        obj("a1", { state: "accepted" }), obj("b2", { state: "auto_accept" }),
        obj("c3", { state: "review" }), obj("d4", { state: "rejected" }),
      ],
    }));
    expect(screen.getByText("1 / 4")).toBeInTheDocument();
    expect(screen.getByText(/1 confirmed/)).toBeInTheDocument();
    expect(screen.getByText(/1 auto/)).toBeInTheDocument();
    expect(screen.getByText(/2 open/)).toBeInTheDocument();
  });
});
