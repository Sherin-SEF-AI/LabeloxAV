import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import { useDialog } from "./useDialog";

function Modal({ open, onClose, children, label = "Test dialog" }: {
  open: boolean; onClose: () => void; children?: React.ReactNode; label?: string;
}) {
  const ref = useDialog<HTMLDivElement>(open, onClose);
  if (!open) return null;
  return (
    <div ref={ref} role="dialog" aria-modal="true" aria-label={label} tabIndex={-1}>
      {children ?? (<><button>first</button><input aria-label="middle" /><button>last</button></>)}
    </div>
  );
}

function Harness({ onClose }: { onClose?: () => void }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button onClick={() => setOpen(true)}>opener</button>
      <button>elsewhere</button>
      <Modal open={open} onClose={() => { onClose?.(); setOpen(false); }} />
    </>
  );
}

describe("useDialog", () => {
  it("moves focus in on open and gives it back on close", async () => {
    render(<Harness />);
    const opener = screen.getByRole("button", { name: "opener" });
    await userEvent.click(opener);
    expect(screen.getByRole("button", { name: "first" })).toHaveFocus();
    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(opener).toHaveFocus();
  });

  it("closes on Escape, once", async () => {
    const onClose = vi.fn();
    render(<Harness onClose={onClose} />);
    await userEvent.click(screen.getByRole("button", { name: "opener" }));
    await userEvent.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("keeps Tab inside, in both directions", async () => {
    render(<Harness />);
    await userEvent.click(screen.getByRole("button", { name: "opener" }));
    const first = screen.getByRole("button", { name: "first" });
    const last = screen.getByRole("button", { name: "last" });
    await userEvent.tab();
    expect(screen.getByLabelText("middle")).toHaveFocus();
    await userEvent.tab();
    expect(last).toHaveFocus();
    await userEvent.tab();
    expect(first).toHaveFocus();
    await userEvent.tab({ shift: true });
    expect(last).toHaveFocus();
  });

  it("focuses the container when there is nothing focusable, and holds Tab there", async () => {
    render(<Modal open onClose={() => {}}><p>just text</p></Modal>);
    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveFocus();
    await userEvent.tab();
    expect(dialog).toHaveFocus();
  });

  it("prefers the element marked data-autofocus", () => {
    render(<Modal open onClose={() => {}}><button>one</button><button data-autofocus>two</button></Modal>);
    expect(screen.getByRole("button", { name: "two" })).toHaveFocus();
  });

  it("leaves focus where React's autoFocus already put it", () => {
    // eslint-disable-next-line jsx-a11y/no-autofocus
    render(<Modal open onClose={() => {}}><button>one</button><input aria-label="named" autoFocus /></Modal>);
    expect(screen.getByLabelText("named")).toHaveFocus();
  });

  it("skips hidden and disabled controls", async () => {
    render(
      <Modal open onClose={() => {}}>
        <button disabled>off</button><button hidden>gone</button><button>only</button>
      </Modal>);
    const only = screen.getByRole("button", { name: "only" });
    expect(only).toHaveFocus();
    await userEvent.tab();
    expect(only).toHaveFocus();
  });

  it("lets only the top dialog answer Escape", async () => {
    const lower = vi.fn();
    const upper = vi.fn();
    function Two() {
      const [top, setTop] = useState(true);
      return (
        <>
          <Modal open onClose={lower} label="lower" />
          <Modal open={top} onClose={() => { upper(); setTop(false); }} label="upper" />
        </>
      );
    }
    render(<Two />);
    await userEvent.keyboard("{Escape}");
    expect(upper).toHaveBeenCalledTimes(1);
    expect(lower).not.toHaveBeenCalled();
    await userEvent.keyboard("{Escape}");
    expect(lower).toHaveBeenCalledTimes(1);
  });

  it("can leave Escape to the caller", async () => {
    const onClose = vi.fn();
    function Sticky() {
      const ref = useDialog<HTMLDivElement>(true, onClose, { closeOnEscape: false });
      return <div ref={ref} role="dialog" aria-label="sticky" tabIndex={-1}><button>ok</button></div>;
    }
    render(<Sticky />);
    await userEvent.keyboard("{Escape}");
    expect(onClose).not.toHaveBeenCalled();
  });

  it("does not pull focus back when closing already moved it elsewhere", async () => {
    function Moves() {
      const [open, setOpen] = useState(false);
      return (
        <>
          <button onClick={() => setOpen(true)}>opener</button>
          <input aria-label="destination" />
          <Modal open={open} onClose={() => setOpen(false)}>
            <button onClick={() => { screen.getByLabelText("destination").focus(); setOpen(false); }}>go</button>
          </Modal>
        </>
      );
    }
    render(<Moves />);
    await userEvent.click(screen.getByRole("button", { name: "opener" }));
    await userEvent.click(screen.getByRole("button", { name: "go" }));
    expect(screen.getByLabelText("destination")).toHaveFocus();
  });
});
