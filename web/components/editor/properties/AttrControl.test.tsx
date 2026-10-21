import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import AttrControl from "./AttrControl";

// A select's option values are strings. The ontology's enums are not always: occlusion is declared as
// [0, 25, 50, 75, 100]. Passing the string back made the server refuse every save of it.
describe("AttrControl enum", () => {
  const occlusion = { type: "enum", values: [0, 25, 50, 75, 100], range: null };

  it("hands back the declared number, not its string", () => {
    const onChange = vi.fn();
    render(<AttrControl name="occlusion" spec={occlusion} value={null} onChange={onChange} />);
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "25" } });
    expect(onChange).toHaveBeenCalledWith(25);
  });

  it("keeps string enums as strings", () => {
    const onChange = vi.fn();
    const direction = { type: "enum", values: ["same", "cross", "wrong_way"], range: null };
    render(<AttrControl name="direction" spec={direction} value={null} onChange={onChange} />);
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "cross" } });
    expect(onChange).toHaveBeenCalledWith("cross");
  });

  it("shows a stored number as selected", () => {
    render(<AttrControl name="occlusion" spec={occlusion} value={50} onChange={() => {}} />);
    expect((screen.getByRole("combobox") as HTMLSelectElement).value).toBe("50");
  });
});
