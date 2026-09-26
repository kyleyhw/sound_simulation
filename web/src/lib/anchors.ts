/**
 * In-page anchors for rendered Markdown (see lib/markdown.ts). Kept free of
 * KaTeX and marked so that main can install the handlers without pulling
 * the Markdown renderer into the main bundle.
 */

/** Scroll the element with this id into view; false when it is not in the page. */
export function scrollToAnchor(id: string): boolean {
  const el = document.getElementById(id);
  if (!el) return false;
  el.scrollIntoView({ block: 'start' });
  return true;
}

/**
 * Scroll every scrolling ancestor of `node` (the app's `.page`, not the
 * window, is the usual scroller) back to the top.
 */
export function scrollPageToTop(node: Element | null): void {
  for (let el = node?.parentElement ?? null; el; el = el.parentElement) {
    if (el.scrollTop !== 0) el.scrollTop = 0;
  }
  if (typeof window !== 'undefined') window.scrollTo(0, 0);
}

let installed = false;

/**
 * Global handlers for rendered Markdown: a click on an in-page anchor link
 * scrolls instead of navigating, and a route with `?h=<id>` scrolls to that
 * id once it appears (documents render asynchronously).
 */
export function installAnchorLinks(): void {
  if (installed || typeof document === 'undefined') return;
  installed = true;
  document.addEventListener('click', (e) => {
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    const a = (e.target as Element | null)?.closest?.('a[data-anchor]');
    if (!a) return;
    e.preventDefault();
    scrollToAnchor(a.getAttribute('data-anchor') ?? '');
  });
  let stop: (() => void) | null = null;
  const follow = () => {
    stop?.();
    stop = null;
    const q = window.location.hash.split('?')[1];
    const id = q ? new URLSearchParams(q).get('h') : null;
    if (!id) return;
    // Wait for the element, then scroll after the page's own layout settles.
    const go = () => requestAnimationFrame(() => requestAnimationFrame(() => scrollToAnchor(id)));
    if (document.getElementById(id)) return go();
    const mo = new MutationObserver(() => {
      if (!document.getElementById(id)) return;
      stop?.();
      go();
    });
    mo.observe(document.body, { childList: true, subtree: true });
    const t = window.setTimeout(() => stop?.(), 10_000);
    stop = () => {
      mo.disconnect();
      window.clearTimeout(t);
      stop = null;
    };
  };
  window.addEventListener('hashchange', follow);
  follow();
}
