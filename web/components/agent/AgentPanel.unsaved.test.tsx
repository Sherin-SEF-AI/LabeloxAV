import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import AgentPanel from "./AgentPanel";

// A box drawn in this session has a local id like `tmp-5` and does not exist on the server, so every
// object-scoped agent operation 404s on it:
//
//   :3000/api/agent/objects/tmp-5/propagate/plan   404 (Not Found)
//
// The panel is given only the id, and `tmp-` is how useEditor marks an unsaved box. The button stays
// enabled because the operation is a reasonable thing to want on the box you just drew - so the panel has
// to say why it cannot, rather than firing a request that fails.

const calls: string[] = [];

vi.mock("@/lib/api", () => {
  const track = (name: string) => vi.fn(async () => { calls.push(name); return {} as never; });
  return {
    api: {
      agentPropagatePlan: track("propagatePlan"),
      agentPropagate: track("propagate"),
      agentCrossCamPlan: track("crossCamPlan"),
      agentCrossCam: track("crossCam"),
      // Everything else the panel touches on mount, stubbed so the render does not throw before the
      // assertion runs. Listed explicitly rather than with a Proxy, so a method added later fails loudly
      // here instead of silently resolving undefined.
      agentSuggest: vi.fn().mockResolvedValue({ suggestions: [] }),
      agentCopilotPattern: vi.fn().mockResolvedValue({ patterns: [] }),
      agentCopilotBatchFix: vi.fn().mockResolvedValue({}),
      agentAttributes: vi.fn().mockResolvedValue({}),
      agentAttributesPlan: vi.fn().mockResolvedValue({ counts: {} }),
      agentCommand: vi.fn().mockResolvedValue({}),
      agentCuboids: vi.fn().mockResolvedValue({}),
      agentCuboidsPlan: vi.fn().mockResolvedValue({ counts: { total: 0 } }),
      agentPlan: vi.fn().mockResolvedValue({}),
      agentReanalyze: vi.fn().mockResolvedValue({}),
      agentReanalyzeAll: vi.fn().mockResolvedValue({}),
      agentReanalyzePlan: vi.fn().mockResolvedValue({ counts: {} }),
      agentRelabel: vi.fn().mockResolvedValue({}),
      agentRelabelPlan: vi.fn().mockResolvedValue({ counts: {} }),
      agentRevert: vi.fn().mockResolvedValue({}),
      agentRun: vi.fn().mockResolvedValue({}),
      agentRunStatus: vi.fn().mockResolvedValue({}),
      opPrecisionAll: vi.fn().mockResolvedValue({ operations: {} }),
    },
    humanizeError: (e: unknown) => String(e),
  };
});
vi.mock("@/lib/toast", () => ({ toast: vi.fn() }));

beforeEach(() => { calls.length = 0; });

async function clickOp(label: RegExp, selectedId: string) {
  render(<AgentPanel frameId="f-1" sessionId="s-1" selectedId={selectedId} />);
  const btn = await screen.findByRole("button", { name: label });
  await userEvent.click(btn);
  return btn;
}

describe("an unsaved box never reaches an object-scoped agent route", () => {
  it("does not call propagate for a tmp- id", async () => {
    await clickOp(/auto-track selected object/i, "tmp-5");
    await waitFor(() => expect(screen.getByText(/save this box first/i)).toBeInTheDocument());
    expect(calls, "no request should be made for an object the server has never seen").toEqual([]);
  });

  it("does not call cross-camera for a tmp- id", async () => {
    await clickOp(/propagate to other cameras/i, "tmp-7");
    await waitFor(() => expect(screen.getByText(/save this box first/i)).toBeInTheDocument());
    expect(calls).toEqual([]);
  });

  it("still runs for a real server id", async () => {
    // The guard must refuse only unsaved boxes, not disable the feature.
    await clickOp(/auto-track selected object/i, "9f3c1e02-0000-4000-8000-000000000000");
    await waitFor(() => expect(calls.length).toBeGreaterThan(0));
    expect(screen.queryByText(/save this box first/i)).not.toBeInTheDocument();
  });
});
