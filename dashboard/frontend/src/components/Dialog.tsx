/**
 * A modal dialog, on the native `<dialog>` element.
 *
 * `showModal()` rather than a hand-rolled overlay, because everything that
 * makes a modal correct is already in the platform: Escape closes it, focus is
 * trapped inside it and restored on close, the page behind it goes inert, and
 * it renders in the top layer so no `z-index` has to be negotiated with the
 * charts. A div with `position: fixed` gets none of that, and the parts people
 * usually skip are the ones a keyboard user needs.
 *
 * Two behaviours worth naming:
 *
 * * **The element is the source of truth for openness.** React tells it to open
 *   or close, but the `close` event tells React -- otherwise Escape would
 *   dismiss the dialog while the state that opened it still said `true`, and
 *   the next click on the trigger would do nothing.
 * * **No entrance animation**, per DESIGN.md. A dialog that fades in is a
 *   dialog you cannot read for 200ms.
 */

import { useEffect, useRef, type ReactNode } from "react";
import { t } from "../lib/i18n";

export interface DialogProps {
  open: boolean;
  title: string;
  onClose: () => void;
  children: ReactNode;
}

export function Dialog(props: DialogProps) {
  const ref = useRef<HTMLDialogElement>(null);
  const { open, onClose } = props;

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    // `open` as an attribute would render it inline and unmodal, so the methods
    // are what drive it. Guarded because calling either twice throws.
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    // Fires for Escape and for `close()` alike, which is what keeps the caller's
    // state honest however the dialog was dismissed.
    const onNativeClose = () => onClose();
    dialog.addEventListener("close", onNativeClose);
    return () => dialog.removeEventListener("close", onNativeClose);
  }, [onClose]);

  return (
    <dialog
      className="dialog"
      ref={ref}
      aria-label={props.title}
      // A click that lands on the element itself landed on the backdrop: the
      // content is inside `.dialog-inner`, so anything hitting the outer box is
      // outside the dialog as the reader sees it.
      onClick={(event) => {
        if (event.target === ref.current) onClose();
      }}
    >
      <div className="dialog-inner">
        <div className="dialog-head">
          <h2 className="dialog-title">{props.title}</h2>
          <button className="btn btn-sm" type="button" onClick={onClose}>
            {t("common.close")}
          </button>
        </div>
        <div className="dialog-body">{props.children}</div>
      </div>
    </dialog>
  );
}
