import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import SessionSelect from "./SessionSelect";

vi.mock("@/lib/api", () => {
  // Declared inside the factory: vi.mock is hoisted above the module body.
  const rows = [
    { session_id: "11111111-1111-1111-1111-111111111111", vehicle_id: "DASHCAM-01", city: "BLR", route: null,
      start_ts_ns: 1_790_000_000_000_000_000, end_ts_ns: 0 },
    { session_id: "22222222-2222-2222-2222-222222222222", vehicle_id: "INDIA-DEL-01", city: "DEL", route: null,
      start_ts_ns: 1_791_000_000_000_000_000, end_ts_ns: 0 },
  ];
  return { api: { sessions: () => Promise.resolve(rows) } };
});

describe("SessionSelect", () => {
  it("finds a drive by what it is and returns its id", async () => {
    const onChange = vi.fn();
    render(<SessionSelect value="" onChange={onChange} />);
    const box = screen.getByRole("combobox");
    await userEvent.click(box);
    await userEvent.type(box, "del");
    await waitFor(() => expect(screen.getAllByRole("option")).toHaveLength(1));
    await userEvent.keyboard("{Enter}");
    expect(onChange).toHaveBeenCalledWith("22222222-2222-2222-2222-222222222222");
  });

  it("a pasted id is taken as-is", async () => {
    const onChange = vi.fn();
    render(<SessionSelect value="" onChange={onChange} />);
    await userEvent.click(screen.getByRole("combobox"));
    await userEvent.paste("11111111-1111-1111-1111-111111111111");
    expect(onChange).toHaveBeenCalledWith("11111111-1111-1111-1111-111111111111");
  });

  it("shows the chosen drive by name, not by id", async () => {
    render(<SessionSelect value="11111111-1111-1111-1111-111111111111" onChange={() => {}} />);
    await waitFor(() => expect((screen.getByRole("combobox") as HTMLInputElement).value).toContain("DASHCAM-01 · BLR"));
  });

  it("says when nothing matches", async () => {
    render(<SessionSelect value="" onChange={() => {}} />);
    const box = screen.getByRole("combobox");
    await userEvent.click(box);
    await userEvent.type(box, "nowhere");
    await waitFor(() => expect(screen.getByText(/No session matches/)).toBeInTheDocument());
  });
});
