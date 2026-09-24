import { useEffect, useMemo, useState } from 'react';
import { assetUrl, docRoute, DOCS, titleOf } from '../lib/docs';
import { renderMarkdown } from '../lib/markdown';
import type { Route } from '../lib/router';

const WRITEUPS = 'docs/writeups/';

/** Research write-ups (plan 10.11, 10.12) and the reports behind them. */
export default function Research({ route }: { route?: Route }) {
  const slug = route?.path.split('/')[2];
  const paths = useMemo(() => Object.keys(DOCS).filter((p) => p.startsWith(WRITEUPS)).sort(), []);
  const reports = useMemo(() => Object.keys(DOCS).filter((p) => p.startsWith('tests/reports/')).sort().reverse(), []);
  const [meta, setMeta] = useState<Record<string, { title: string; sub: string }>>({});
  const [html, setHtml] = useState<string | null>(null);

  useEffect(() => {
    void Promise.all(paths.map(async (p) => [p, await DOCS[p]()] as const)).then((all) =>
      setMeta(
        Object.fromEntries(
          all.map(([p, md]) => {
            const sub = md.match(/^\*(.+)\*$/m)?.[1] ?? '';
            const lede = md.split('\n\n').find((b) => !b.startsWith('#') && !b.startsWith('*')) ?? '';
            return [p, { title: titleOf(md, p), sub: sub || lede.slice(0, 160) }];
          }),
        ),
      ),
    );
  }, [paths]);

  useEffect(() => {
    setHtml(null);
    if (!slug) return;
    const path = `${WRITEUPS}${slug}.md`;
    const load = DOCS[path];
    if (!load) return setHtml('<p>No such write-up.</p>');
    void load().then((md) => {
      setHtml(renderMarkdown(md, { path, resolveAsset: assetUrl, routeFor: (p) => (p.startsWith(WRITEUPS) ? `/research/${p.slice(WRITEUPS.length, -3)}` : docRoute(p)) }));
      window.scrollTo(0, 0);
    });
  }, [slug]);

  if (slug) {
    return (
      <div className="content prose" data-testid="writeup">
        <p className="muted" style={{ fontSize: 13 }}>
          <a href="#/research">Research</a>
        </p>
        {html === null ? <p className="muted">Loading…</p> : <div dangerouslySetInnerHTML={{ __html: html }} />}
      </div>
    );
  }

  return (
    <div className="content" data-testid="research">
      <h1>Research</h1>
      <p className="lede">
        What the project has found so far, negative results included. Each write-up links to the report and the code behind its numbers.
      </p>
      <div className="card-grid">
        {paths.map((p) => (
          <a key={p} className="card" href={`#/research/${p.slice(WRITEUPS.length, -3)}`} data-testid="writeup-card">
            <h3 style={{ margin: '2px 0 6px' }}>{meta[p]?.title ?? '…'}</h3>
            <p className="muted" style={{ margin: 0, fontSize: 13.5 }}>
              {meta[p]?.sub}
            </p>
          </a>
        ))}
      </div>
      <h2 style={{ fontSize: 18, marginTop: 28 }}>Reports</h2>
      <ul>
        {reports.map((p) => (
          <li key={p}>
            <a href={`#${docRoute(p)}`}>{p.replace('tests/reports/', '').replace(/\.md$/, '').replace(/_/g, ' ')}</a>
          </li>
        ))}
      </ul>
    </div>
  );
}
