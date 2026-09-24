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
