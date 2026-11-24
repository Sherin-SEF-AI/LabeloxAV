import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it } from "vitest";

import ConfirmProvider, { useConfirm } from "./ConfirmProvider";

function Asker() {
  const confirm = useConfirm();
  const [answer, setAnswer] = useState("unasked");
  return (
    <>
      <button onClick={async () => setAnswer(String(await confirm({ title: "Remove webhook?", body: "No undo.", danger: true })))}>
        ask
      </button>
      <output>{answer}</output>
    </>
  );
}

const setup = async () => {
  render(<ConfirmProvider><Asker /></ConfirmProvider>);
  await userEvent.click(screen.getByRole("button", { name: "ask" }));
  return screen.getByRole("alertdialog", { name: "Remove webhook?" });
};

describe("the confirm dialog", () => {
  it("is a labelled, described modal with focus on Cancel", async () => {
    const dialog = await setup();
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(dialog).toHaveAccessibleDescription("No undo.");
    expect(screen.getByRole("button", { name: "Cancel" })).toHaveFocus();
  });

  it("Enter on the focused Cancel button cancels", async () => {
    // It used to confirm: the dialog caught Enter before the button did, so the safe default was not safe.
    await setup();
    await userEvent.keyboard("{Enter}");
    expect(screen.getByRole("status")).toHaveTextContent("false");
  });

  it("Enter on Confirm confirms, and so does Enter with no button focused", async () => {
    await setup();
    await userEvent.tab();
    expect(screen.getByRole("button", { name: "Confirm" })).toHaveFocus();
    await userEvent.keyboard("{Enter}");
    expect(screen.getByRole("status")).toHaveTextContent("true");

    await userEvent.click(screen.getByRole("button", { name: "ask" }));
    screen.getByRole("alertdialog").focus();
    await userEvent.keyboard("{Enter}");
    expect(screen.getByRole("status")).toHaveTextContent("true");
  });

  it("Escape cancels and gives focus back to what asked", async () => {
    await setup();
    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("false");
    expect(screen.getByRole("button", { name: "ask" })).toHaveFocus();
  });

  it("keeps Tab between its two buttons", async () => {
    await setup();
    await userEvent.tab();
    await userEvent.tab();
    expect(screen.getByRole("button", { name: "Cancel" })).toHaveFocus();
  });
});
