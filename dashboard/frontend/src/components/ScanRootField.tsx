/**
 * The scan root, and the one control that changes it.
 *
 * The root is the single most common thing to get wrong about a dashboard --
 * an empty run list is almost always a mistyped path or a root one level off --
 * and until now fixing it meant restarting the server with a different
 * argument. This makes it an edit in the place that already reports the
 * problem.
 *
 * Three rules the shape follows:
 *
 * * **Read-only until asked.** The path is a code span, not a permanently
 *   focusable input. This row is read on every visit and edited almost never,
 *   and an input box invites a click on the one control that empties the page.
 * * **The server's answer is what gets rendered**, never an optimistic guess.
 *   A rejected path leaves the old root on screen because the old root is still
 *   the one being scanned.
 * * **No control when the server forbids it.** `editable: false` renders the
 *   plain row, for the same reason the metrics view hides the rank drill-down
 *   on a run without per-worker data: a control that can only fail is a
 *   question about a feature rather than an answer about this server.
 */

import { useState } from "react";
import type { ScanRoot } from "../api/types";
import { api, ApiError } from "../api/client";
import { t } from "../lib/i18n";
import { Badge, Code } from "./primitives";

export interface ScanRootFieldProps {
  root: ScanRoot;
  /** Called with the server's new state of the root, never with a guess. */
  onChanged: (root: ScanRoot) => void;
}

export function ScanRootField(props: ScanRootFieldProps) {
  const { root } = props;
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(root.path);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const open = () => {
    setDraft(root.path);
    setError(null);
    setEditing(true);
  };

  const close = () => {
    setEditing(false);
    setError(null);
  };

  /**
   * Submit a path, or `null` to reset.
   *
   * Both gestures go through one function because both are the same request and
   * both fail the same way -- a locked server refuses a reset too.
   */
  const submit = async (path: string | null) => {
    setBusy(true);
    setError(null);
    try {
      props.onChanged(await api.setScanRoot(path));
      setEditing(false);
    } catch (cause) {
      // The server's own sentence, verbatim: it is the only party that knows
      // why, and it names the path it could not find. `detail` rather than
      // `message`, which carries an "HTTP 400: " prefix that tells the operator
      // nothing they can act on.
      setError(
        cause instanceof ApiError
          ? cause.detail
          : cause instanceof Error
            ? cause.message
            : String(cause),
      );
    } finally {
      setBusy(false);
    }
  };

  if (!editing) {
    return (
      <div className="scan-root">
        <Code title={root.path}>{root.path}</Code>
        {/* Neither badge is a count. They separate the two ways an empty
            dashboard happens, which is the whole reason this row exists. */}
        {!root.exists ? (
          <Badge tone="unreachable">{t("server.missing")}</Badge>
        ) : root.run_count === 0 ? (
          <Badge tone="unknown">{t("server.noRunsFound")}</Badge>
        ) : null}
        {root.editable && (
          <button className="btn btn-sm" type="button" onClick={open}>
            {t("server.scanRootChange")}
          </button>
        )}
        {root.editable && !root.is_default && (
          <button
            className="btn btn-sm"
            type="button"
            disabled={busy}
            title={t("server.scanRootResetTitle", { path: root.default_path })}
            onClick={() => void submit(null)}
          >
            {t("server.scanRootReset")}
          </button>
        )}
        {error && <span className="scan-root-error">{error}</span>}
      </div>
    );
  }

  return (
    <form
      className="scan-root"
      onSubmit={(event) => {
        event.preventDefault();
        void submit(draft);
      }}
    >
      <input
        className="scan-root-input"
        type="text"
        value={draft}
        autoFocus
        spellCheck={false}
        autoComplete="off"
        disabled={busy}
        aria-label={t("server.scanRootLabel")}
        placeholder={t("server.scanRootPlaceholder")}
        onChange={(event) => setDraft(event.target.value)}
        // Escape leaves without changing anything. A dialog you can only leave
        // by submitting it is how a wrong path gets submitted.
        onKeyDown={(event) => {
          if (event.key === "Escape") close();
        }}
      />
      <button className="btn btn-sm btn-primary" type="submit" disabled={busy}>
        {t("server.scanRootSave")}
      </button>
      <button className="btn btn-sm" type="button" disabled={busy} onClick={close}>
        {t("server.scanRootCancel")}
      </button>
      {error && <span className="scan-root-error">{error}</span>}
    </form>
  );
}
