import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { pressable } from "./pressable";

describe("pressable", () => {
  it("is a focusable button that a click, Enter and Space all activate", async () => {
    const onPress = vi.fn();
    render(<div {...pressable(onPress)}>row</div>);
    const row = screen.getByRole("button", { name: "row" });
    await userEvent.tab();
    expect(row).toHaveFocus();
    await userEvent.click(row);
    await userEvent.keyboard("{Enter}");
    await userEvent.keyboard(" ");
    expect(onPress).toHaveBeenCalledTimes(3);
  });

  it("ignores other keys, and keys typed in something inside it", async () => {
    const onPress = vi.fn();
    render(<div {...pressable(onPress)}>row<input aria-label="inner" /></div>);
    screen.getByRole("button").focus();
    await userEvent.keyboard("a");
    screen.getByLabelText("inner").focus();
    await userEvent.keyboard("{Enter}");
    expect(onPress).not.toHaveBeenCalled();
  });

  it("does nothing, and leaves the tab order, when disabled", async () => {
    const onPress = vi.fn();
    render(<div {...pressable(onPress, { disabled: true })}>row</div>);
    const row = screen.getByRole("button");
    expect(row).toHaveAttribute("aria-disabled", "true");
    expect(row).toHaveAttribute("tabindex", "-1");
    await userEvent.click(row);
    row.focus();
    await userEvent.keyboard("{Enter}");
    expect(onPress).not.toHaveBeenCalled();
  });
});
