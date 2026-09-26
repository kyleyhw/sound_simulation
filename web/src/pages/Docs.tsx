import { ChevronDown } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { assetUrl, docPathFromRoute, docRoute, DOCS, scrollPageTop } from '../lib/docs';
import { renderMarkdown } from '../lib/markdown';
import type { Route } from '../lib/router';

/** Module-level notes for the Python package: useful to contributors, noise to everyone else. */
const DEVELOPER = new Set(
  ['calculate', 'data_io', 'main', 'setup', 'utils', 'visualize', 'simulate', 'waveforms', 'gpu', 'index', 'plan_audit'].map((n) => `docs/${n}.md`),
);
const START = ['README.md', 'CURRENT_STATE.md', 'docs/web_app.md'];

interface Group {
  title: string;
  /** Collapsed unless it holds the open document. */
  collapsed?: boolean;
  match: (p: string) => boolean;
}

const GROUPS: Group[] = [
  { title: 'Start here', match: (p) => START.includes(p) || p.startsWith('docs/writeups/') },
  { title: 'Guides', match: (p) => p === 'PROJECT_PLAN.md' || (p.startsWith('docs/') && !p.startsWith('docs/writeups/') && !START.includes(p) && !DEVELOPER.has(p)) },
  { title: 'Developer reference', collapsed: true, match: (p) => DEVELOPER.has(p) },
  { title: 'Reports', collapsed: true, match: (p) => p.startsWith('tests/reports/') },
];

/** Friendly names for the documents a newcomer is pointed at; the rest use their file name. */
const LABELS: Record<string, string> = {
  'README.md': 'Overview',
  'CURRENT_STATE.md': 'Current state',
  'PROJECT_PLAN.md': 'Project plan',
  'docs/web_app.md': 'The web app',
  'docs/writeups/is_the_simulator_right.md': 'Is the simulator right?',
  'docs/writeups/real_rooms.md': 'Measuring a real room',
  'docs/writeups/shaping_sound.md': 'Shaping sound with speakers',
  'docs/writeups/what_can_a_laptop_hear.md': 'What can a laptop hear?',
  'docs/benchmark.md': 'Room-sensing benchmark',
  'docs/control.md': 'Sound-field control',
  'docs/demos.md': 'Demonstrations',
  'docs/imaging.md': 'Room imaging (no ML)',
  'docs/lab.md': 'Lab: real hardware',
  'docs/learning.md': 'Learned sensing (Phase 2)',
  'docs/physics.md': 'Physics and verification',
  'docs/index.md': 'index',
};

const label = (p: string) =>
  LABELS[p] ??
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
  // Phones: the document list folds away once a document is chosen (B31).
  const [listOpen, setListOpen] = useState(false);
  const paths = useMemo(() => Object.keys(DOCS).sort(), []);
  const grouped = useMemo(() => {
    const seen = new Set<string>();
    return GROUPS.map((g) => {
      const items = paths.filter((p) => !seen.has(p) && g.match(p));
      items.forEach((p) => seen.add(p));
      if (g.title === 'Start here') items.sort((a, b) => (START.indexOf(a) + 1 || 99) - (START.indexOf(b) + 1 || 99) || a.localeCompare(b));
      if (g.title === 'Reports') items.reverse();
      return { ...g, items };
    }).filter((g) => g.items.length);
  }, [paths]);

  useEffect(() => {
    let live = true;
    const load = DOCS[path];
    setHtml(null);
    setMissing(!load);
    setListOpen(false);
    scrollPageTop();
    if (load)
      void load().then((md) => {
        if (!live) return;
        setHtml(renderMarkdown(md, { path, resolveAsset: assetUrl, routeFor: docRoute }));
        scrollPageTop();
      });
    return () => {
      live = false;
    };
  }, [path]);

  return (
    <div className="docs" data-testid="docs">
      <nav className={`docs-nav${listOpen ? ' open' : ''}`} aria-label="Documents">
        <button className="btn docs-nav-toggle" aria-expanded={listOpen} onClick={() => setListOpen((o) => !o)} data-testid="docs-list-toggle">
          All documents <ChevronDown size={15} aria-hidden />
        </button>
        <div className="docs-nav-list">
          {grouped.map((g) => {
            const list = (
              <ul>
                {g.items.map((p) => (
                  <li key={p}>
                    <a href={`#${docRoute(p)}`} aria-current={p === path ? 'page' : undefined}>
                      {label(p)}
                    </a>
                  </li>
                ))}
              </ul>
            );
            return g.collapsed ? (
              <details key={g.title} open={g.items.includes(path) || undefined} data-testid={`docs-group-${g.title.split(' ')[0].toLowerCase()}`}>
                <summary>
                  <h3>{g.title}</h3>
                </summary>
                {list}
              </details>
            ) : (
              <div key={g.title}>
                <h3>{g.title}</h3>
                {list}
              </div>
            );
          })}
        </div>
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
