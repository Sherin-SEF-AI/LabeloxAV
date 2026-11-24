import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { FrameMeta, Ontology } from "@/lib/types";
import PropertiesPanel, { type PropertiesPanelProps } from "./PropertiesPanel";
import type { EdObject, EditorState } from "../useEditor";

// The right rail on a view-only frame. The agent, bulk edit and the frame-data tools write to the frame,
// most of them straight to the server, so none of them is offered; and because the lock can be lost
// halfway through using one, taking them away must not unmount them.

vi.mock("@/lib/colors", () => ({ classColor: () => "#888" }));
vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }) }));

let agentMounts = 0;
vi.mock("@/components/agent/AgentPanel", async () => {
  const { useEffect } = await import("react");
  function MockAgentPanel() {
    useEffect(() => { agentMounts += 1; }, []);
    return <div data-testid="agent">agent panel</div>;
  }
  return { default: MockAgentPanel };
});
vi.mock("@/components/agent/BulkEditBar", () => ({ default: () => <div data-testid="bulk">bulk edit bar</div> }));
vi.mock("@/components/editor/SceneGraphPanel", () => ({ default: () => <div data-testid="scene">scene graph</div> }));
vi.mock("@/components/editor/HistoryPanel", () => ({ default: () => <div data-testid="history">history</div> }));

const ONTO = {
  classes: [{ id: 1, name: "sedan", l0: "vehicle", l1: "car", india: false }],
  attributes: {}, attribute_scope: {},
} as unknown as Ontology;
const META = { session_id: "s1", is_lidar: false } as unknown as FrameMeta;

const OBJ: EdObject = {
  id: "aaaaaaaa11", class_id: 1, class_name: "sedan", bbox: [0, 0, 10, 10], mask: [], attrs: {},
  conf: 0.9, state: "review", visible: true,
};
const ST: EditorState = {
  objects: [OBJ], deleted: [], selectedId: null, selectedIds: [], tool: "select",
  viewport: { scale: 1, ox: 0, oy: 0 }, candidate: null, touched: [], past: [], future: [],
};

function panel(readOnly: string | null, selected: EdObject | null = null) {
  const props: PropertiesPanelProps = {
    frame: { id: "f1", meta: META, onto: ONTO, dirty: false, flash: vi.fn(), readOnly },
    editor: { st: ST, dispatch: vi.fn(), selected },
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

const NOTE = "Being edited by Asha — view only";

// The open tab is remembered in localStorage, so one test's tab would be the next one's starting point.
beforeEach(() => { window.localStorage.clear(); });

describe("the tools on a view-only frame", () => {
  it("are replaced by the reason, in every tab", async () => {
    const user = userEvent.setup();
    render(panel(NOTE));
    expect(screen.getByText(new RegExp(NOTE))).toBeVisible();
    expect(screen.getByTestId("agent")).not.toBeVisible();

    await user.click(screen.getByRole("tab", { name: "bulk edit" }));
    expect(screen.getByTestId("bulk")).not.toBeVisible();
    await user.click(screen.getByRole("tab", { name: "frame data" }));
    expect(screen.getByTestId("history")).not.toBeVisible();
    // Said once per tab, in whichever tab is open.
    expect(screen.getAllByText(new RegExp(NOTE)).filter((el) => el.closest("[hidden]") === null)).toHaveLength(1);
  });

  it("are shown as usual on an editable frame", () => {
    render(panel(null));
    expect(screen.getByTestId("agent")).toBeVisible();
    expect(screen.queryByText(/view only/)).toBeNull();
  });

  it("stay mounted when the lock is lost and come back when it returns", () => {
    agentMounts = 0;
    const { rerender } = render(panel(null));
    expect(agentMounts).toBe(1);
    rerender(panel(NOTE));
    expect(screen.getByTestId("agent")).not.toBeVisible();
    rerender(panel(null));
    expect(screen.getByTestId("agent")).toBeVisible();
    // One mount throughout: whatever was typed into it is still there.
    expect(agentMounts).toBe(1);
  });

  it("still lists the objects, so a viewer can read the frame", () => {
    render(panel(NOTE));
    expect(screen.getAllByText("sedan").length).toBeGreaterThan(0);
  });

  it("disables the two propagate buttons on the selected object, which write from the card itself", () => {
    const { rerender } = render(panel(null, OBJ));
    const forward = screen.getByRole("button", { name: /propagate forward 12 frames/ });
    const both = screen.getByRole("button", { name: /propagate both ways/ });
    expect(forward).toBeEnabled();
    expect(both).toBeEnabled();
    rerender(panel(NOTE, OBJ));
    expect(screen.getByRole("button", { name: /propagate forward 12 frames/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: /propagate both ways/ })).toBeDisabled();
  });
});
