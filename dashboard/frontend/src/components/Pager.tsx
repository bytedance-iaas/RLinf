/**
 * The pager, shared by every paginated list.
 *
 * One component because there is one right answer to "which page am I on", and
 * two lists that need it. The run list and the event log had no reason to
 * disagree about the shape of it, and before this they did: the event log wrote
 * its own, with a field label reading `Page` in front of the buttons and the
 * position between them as a bare `1 / 3`.
 *
 * Two things that shape is careful about:
 *
 * * **The position reads as a sentence, in the middle.** `Page 1 of 3` between
 *   the arrows, not a `Page` label glued to the end of the filter row followed
 *   by a fraction. Beside a `1-50 of 132` readout, a second bare fraction is one
 *   number pair too many to hold in the head, and the label attached itself to
 *   whatever field happened to precede it.
 * * **It renders nothing when there is one page.** A list that fits on a page
 *   has no navigation to offer, and a disabled pager is furniture that says
 *   there might be more somewhere.
 */

import { t } from "../lib/i18n";

/**
 * Rows per page, for every list in the console.
 *
 * Twenty is what fits above the fold on a laptop without scrolling past the
 * list into whatever follows it, which is the geometry both lists are read in:
 * the run list answers "is anything wrong", and the event log is read after an
 * alert. Neither answer should need a scroll to know it has been fully asked.
 *
 * One constant because it is one decision. The event log paged at fifty until
 * the run list picked twenty, and a console where two lists disagree about how
 * much a page holds makes the reader relearn the pager on every view.
 */
export const PAGE_SIZE = 20;

export interface PagerProps {
  /** Zero-based, because the call sites slice arrays with it. */
  page: number;
  pageCount: number;
  onChange: (page: number) => void;
}

export function Pager(props: PagerProps) {
  const { page, pageCount } = props;
  if (pageCount <= 1) return null;

  const first = page === 0;
  const last = page >= pageCount - 1;

  return (
    <nav className="pager" aria-label={t("pager.label")}>
      <button
        className="btn"
        type="button"
        onClick={() => props.onChange(0)}
        disabled={first}
        aria-label={t("pager.first")}
      >
        ‹‹
      </button>
      <button
        className="btn"
        type="button"
        onClick={() => props.onChange(page - 1)}
        disabled={first}
        aria-label={t("pager.prev")}
      >
        ‹
      </button>
      {/* Announced on change, because for a screen reader the only evidence
          that the button did anything is this number. */}
      <span className="pager-position" aria-live="polite">
        {t("pager.page", { current: page + 1, total: pageCount })}
      </span>
      <button
        className="btn"
        type="button"
        onClick={() => props.onChange(page + 1)}
        disabled={last}
        aria-label={t("pager.next")}
      >
        ›
      </button>
      <button
        className="btn"
        type="button"
        onClick={() => props.onChange(pageCount - 1)}
        disabled={last}
        aria-label={t("pager.last")}
      >
        ››
      </button>
    </nav>
  );
}
