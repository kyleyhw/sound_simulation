/** Tiny hash router: "#/route?query". Works on static hosting (GitHub Pages). */

import { useEffect, useState } from 'react';

export interface Route {
  path: string;
  query: URLSearchParams;
}

export function parseHash(hash: string): Route {
  const h = hash.replace(/^#/, '');
  const [path, q] = h.split('?');
  return { path: path || '/', query: new URLSearchParams(q ?? '') };
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));
  useEffect(() => {
    const on = () => setRoute(parseHash(window.location.hash));
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  return route;
}

export function navigate(path: string): void {
  window.location.hash = path;
}

export type PageId = 'home' | 'sandbox' | 'echo' | 'gallery' | 'learn' | 'research' | 'loop' | 'lab' | 'docs' | 'notfound';

/** Pages that take sub-paths (#/learn/<id>, #/research/<slug>, #/docs/<path>). */
const NESTED: PageId[] = ['learn', 'research', 'docs'];
const FLAT: PageId[] = ['sandbox', 'echo', 'gallery', 'loop', 'lab'];

/**
 * Which page a route shows. The sandbox lives at #/sandbox; the old share
 * links (#/?s=… and #/?preset=…) still open it, and #/ alone is the home page.
 */
export function pageOf(route: Route): PageId {
  const parts = route.path.replace(/\/+$/, '').split('/').filter(Boolean);
  if (parts.length === 0) return route.query.has('s') || route.query.has('preset') ? 'sandbox' : 'home';
  const head = parts[0] as PageId;
  if (NESTED.includes(head)) return head;
  if (FLAT.includes(head) && parts.length === 1) return head;
  return 'notfound';
}
