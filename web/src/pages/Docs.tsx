import { useEffect, useMemo, useState } from 'react';
import { assetUrl, docPathFromRoute, docRoute, DOCS } from '../lib/docs';
import { renderMarkdown } from '../lib/markdown';
import type { Route } from '../lib/router';

const GROUPS: { title: string; match: (p: string) => boolean }[] = [
  { title: 'Project', match: (p) => !p.includes('/') },
  { title: 'Guides and reference', match: (p) => p.startsWith('docs/') },
  { title: 'Reports', match: (p) => p.startsWith('tests/reports/') },
];

const label = (p: string) =>
  p
    .replace(/^docs\//, '')
    .replace(/^tests\/reports\//, '')
    .replace(/\.md$/, '')
    .replace(/_/g, ' ');

/** Docs site (plan 10.9): every Markdown document in the repo, rendered with maths. */
export default function Docs({ route }: { route: Route }) {
  const path = docPathFromRoute(route.path);
  const [html, setHtml] = useState<string | null>(null);
  const [missing, setMissing] = useState(false);
  const paths = useMemo(() => Object.keys(DOCS).sort(), []);

  useEffect(() => {
    let live = true;
    const load = DOCS[path];
    setHtml(null);
    setMissing(!load);
    if (load)
      void load().then((md) => {
        if (!live) return;
        setHtml(renderMarkdown(md, { path, resolveAsset: assetUrl, routeFor: docRoute }));
        window.scrollTo(0, 0);
      });
    return () => {
      live = false;
    };
  }, [path]);

  return (
    <div className="docs" data-testid="docs">
      <nav className="docs-nav" aria-label="Documents">
        {GROUPS.map((g) => (
          <div key={g.title}>
            <h3>{g.title}</h3>
            <ul>
              {paths.filter(g.match).map((p) => (
                <li key={p}>
                  <a href={`#${docRoute(p)}`} aria-current={p === path ? 'page' : undefined}>
                    {label(p)}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>
      <article className="prose" data-testid="doc-body">
        {missing ? <p>No such document: {path}</p> : html === null ? <p className="muted">Loading…</p> : <div dangerouslySetInnerHTML={{ __html: html }} />}
        <p className="muted" style={{ fontSize: 12.5, marginTop: 32 }}>
          Source:{' '}
          <a href={`https://github.com/kyleyhw/sound_simulation/blob/main/${path}`} target="_blank" rel="noreferrer">
            {path}
          </a>
        </p>
      </article>
    </div>
  );
}
