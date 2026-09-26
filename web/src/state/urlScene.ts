/**
 * Scenes opened from the URL (`#/sandbox?preset=…` or `?s=…`) are loaded
 * once per history entry. The entry is marked in `history.state` when it is
 * loaded, so browser Back/Forward onto it later does not reload the preset
 * over the user's edits. The mark carries a per-page-load session id: after
 * a reload (fresh store, the edits are gone anyway) the scene loads again.
 */

const SESSION = Math.random().toString(36).slice(2);

interface Mark {
  sandboxQuery: string;
  sandboxSession: string;
}

function currentState(): Record<string, unknown> {
  const st = typeof history !== 'undefined' ? (history.state as unknown) : null;
  return st && typeof st === 'object' ? (st as Record<string, unknown>) : {};
}

/** True when this history entry's URL scene was already loaded in this page session. */
export function urlSceneConsumed(query: string): boolean {
  const st = currentState() as Partial<Mark>;
  return st.sandboxSession === SESSION && st.sandboxQuery === query;
}

/** History state that marks `query` as loaded (for replaceState/pushState). */
export function consumedState(query: string): Record<string, unknown> {
  return { ...currentState(), sandboxQuery: query, sandboxSession: SESSION } satisfies Record<string, unknown> & Mark;
}

/** Mark the current history entry's URL scene as loaded. */
export function markUrlSceneConsumed(query: string): void {
  try {
    history.replaceState(consumedState(query), '');
  } catch {
    /* history unavailable (sandboxed iframe): Back may reload the scene */
  }
}
