"use client";

// The frame editor's right rail, for the object-annotation modes.
//
// It used to be two exclusive tabs, "objects" and "tools", each a long scroll of always-expanded sections,
// assembled inline inside a 2,318-line page behind seven separate `rightTab === "..."` guards. Two things
// were wrong with that beyond the length. Objects and tools were mutually exclusive, so you could see the
// object you were labelling or the agent labelling it and never both. And the tools tab stacked five open
// sections, which is not a hierarchy, so the one being used sat wherever it happened to be declared.
//
// Now: the objects list is always present, and the tools are three sub-tabs beneath it.
//
// KEEP-ALIVE. Agent, bulk edit and the frame-data sections all shared one tab before, so moving between
// them never unmounted anything. Splitting them across three tabs makes a naive conditional render throw
// away an in-flight dry-run plan, a typed prompt and the reanalysis findings on every switch. So a group
// mounts on first visit and then stays mounted, hidden with the `hidden` attribute, which takes it out of
// both paint and the accessibility tree. Mounting all three up front instead would fire four requests on
// every panel open for tabs nobody has clicked.

import { useEffect, useRef, useState, type Dispatch } from "react";

import type { FrameMeta, LaneRow, ObjectDynamicsRow, Ontology, OntologyClass, Relationship } from "@/lib/types";
import AgentPanel from "@/components/agent/AgentPanel";
import BulkEditBar from "@/components/agent/BulkEditBar";
import SceneGraphPanel from "@/components/editor/SceneGraphPanel";
import HistoryPanel from "@/components/editor/HistoryPanel";
import PanelSection from "@/components/editor/PanelSection";

import DynamicsCard from "./DynamicsCard";
import LidarBevSection from "./LidarBevSection";
import LintCard from "./LintCard";
import ObjectsCard from "./ObjectsCard";
import PanelHeader from "./PanelHeader";
import RoadSegSection from "./RoadSegSection";
import SelectedObjectCard from "./SelectedObjectCard";
import SelectionFilters from "./SelectionFilters";
import ToolTabs from "./ToolTabs";
import { PREF_TOOL_TAB, usePanelPref } from "./panelPrefs";
import type { Action, EdObject, EditorState } from "../useEditor";

const TABS = [
  { key: "agent", label: "agent" },
  { key: "bulk", label: "bulk edit" },
  { key: "frame", label: "frame data" },
] as const;
type ToolTab = (typeof TABS)[number]["key"];
const TAB_KEYS = TABS.map((t) => t.key);

export type PropertiesPanelProps = {
  frame: { id: string; meta: FrameMeta; onto: Ontology; dirty: boolean; flash: (m: string) => void };
  editor: { st: EditorState; dispatch: Dispatch<Action>; selected: EdObject | null };
  /** What the guideline check said about this frame, and which rules could not run. */
  lint?: import("./LintCard").LintResult | null;
  onOpenLintIssues?: () => void;
  /** The frame's risk order, marked on the object rows and never used to reorder them. */
  risk?: { rank: Record<string, number>; score: Record<string, { risk: number; reasons: string[] }>;
           coverage: number } | null;
  klass: { current: OntologyClass | null; onPick: (c: OntologyClass) => void; onAdd: (raw: string) => void };
  sel: {
    relationships: Relationship[];
    linkKind: string;
    linkFrom: string | null;
    dynamics: Record<string, ObjectDynamicsRow>;
    onSetAttr: (name: string, val: unknown) => void;
    onLinkKind: (k: string) => void;
    onToggleLink: () => void;
    onDeleteRelationship: (rid: string) => void;
    onRecomputeDynamics: () => void;
    /** Lift this object's 2D box to a cuboid with the monocular solve. */
    onFitCuboid?: () => void;
    /** Set the cuboid's yaw, in radians. */
    onSetYaw?: (yaw: number) => void;
  };
  onCollapse: () => void;
  tools: {
    lanes: LaneRow[];
    hasDrivable: boolean;
    onSegRoad: () => void;
    onProposeLanes: () => void;
    onAgentApplied: () => void;
    onHistoryRestored: () => void;
    onLiftCuboids: () => void | Promise<void>;
  };
};

export default function PropertiesPanel({ frame, editor, klass, sel, tools, onCollapse, risk, lint, onOpenLintIssues }: PropertiesPanelProps) {
  const { st, dispatch, selected } = editor;
  const [tab, setTab] = usePanelPref<ToolTab>(PREF_TOOL_TAB, "agent", TAB_KEYS);

  // Which groups have ever been shown. A ref alongside the state because the effect below has to read the
  // current set without re-running on every change to it.
  const [visited, setVisited] = useState<Set<ToolTab>>(() => new Set<ToolTab>(["agent"]));
  const visitedRef = useRef(visited);
  visitedRef.current = visited;
  useEffect(() => {
    if (visitedRef.current.has(tab)) return;
    setVisited((s) => new Set(s).add(tab));
  }, [tab]);

  const panel = (key: ToolTab, children: React.ReactNode) => {
    if (!visited.has(key)) return null;
    return (
      <div key={key} role="tabpanel" hidden={key !== tab} tabIndex={0}
        id={`props-panel-${key}`} aria-labelledby={`props-tab-${key}`}>
        {children}
      </div>
    );
  };

  return (
    <>
      <PanelHeader objects={st.objects} dirty={frame.dirty} selectedName={selected?.class_name ?? null}
        currentClass={klass.current} classes={frame.onto.classes}
        onPickClass={klass.onPick} onAddClass={klass.onAdd} onCollapse={onCollapse} />

      <div className="flex-1 min-h-0 overflow-y-auto">
        {selected && (
          <>
            <div className="border-b hairline">
              <SelectedObjectCard object={selected} onto={frame.onto}
                relationships={sel.relationships} linkKind={sel.linkKind} linkFrom={sel.linkFrom}
                onLinkKind={sel.onLinkKind} onToggleLink={sel.onToggleLink}
                onDeleteRelationship={sel.onDeleteRelationship} onSetAttr={sel.onSetAttr}
                onFitCuboid={sel.onFitCuboid} onSetYaw={sel.onSetYaw} />
            </div>
            <PanelSection title="dynamics">
              <DynamicsCard row={sel.dynamics[selected.id]} onRecompute={sel.onRecomputeDynamics} />
            </PanelSection>
          </>
        )}

        <ObjectsCard objects={st.objects} selectedIds={st.selectedIds} dispatch={dispatch}
          filters={<SelectionFilters dispatch={dispatch} />} risk={risk} />

        <LintCard lint={lint ?? null} selectedIds={st.selectedIds}
          onSelect={(oid) => dispatch({ t: "select", id: oid })}
          onOpenIssues={onOpenLintIssues} />

        <ToolTabs tabs={[...TABS]} value={tab} onChange={setTab} idPrefix="props" label="Editor tools" />

        {panel("agent",
          <AgentPanel frameId={frame.id} sessionId={frame.meta.session_id ?? null}
            selectedId={st.selectedId} onApplied={tools.onAgentApplied} embedded />)}

        {panel("bulk",
          <BulkEditBar frameId={frame.id} sessionId={frame.meta.session_id}
            onApplied={() => frame.flash("bulk edit applied (routed to review)")} embedded />)}

        {panel("frame", (
          <>
            {frame.meta.is_lidar && (
              <PanelSection title="lidar bev" storageKey="lidar-bev"
                badge={`${(frame.meta.lidar_points ?? 0).toLocaleString()} pts`}>
                <LidarBevSection onLift={tools.onLiftCuboids} />
              </PanelSection>
            )}
            <PanelSection title="history and saves" storageKey="history" defaultOpen
              badge={`${st.past.length} step${st.past.length === 1 ? "" : "s"}`}>
              <HistoryPanel frameId={frame.id} past={st.past} future={st.future}
                onJump={(at) => dispatch({ t: "jump", at })}
                onRestored={tools.onHistoryRestored} flash={frame.flash} />
            </PanelSection>
            <PanelSection title="scene graph + vlm dataset" storageKey="scene-graph">
              <SceneGraphPanel frameId={frame.id} embedded />
            </PanelSection>
            <PanelSection title="road segmentation" storageKey="road-seg"
              badge={`${tools.lanes.length} lanes${tools.hasDrivable ? " · drivable" : ""}`}>
              <RoadSegSection frameId={frame.id}
                onSegRoad={tools.onSegRoad} onProposeLanes={tools.onProposeLanes} />
            </PanelSection>
          </>
        ))}
      </div>
    </>
  );
}
