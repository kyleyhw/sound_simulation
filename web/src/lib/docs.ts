/**
 * The published documents (docs site, write-ups): the repository's Markdown,
 * bundled at build time and loaded on demand per page.
 */

const TEXTS = import.meta.glob(['../../../docs/**/*.md', '../../../tests/reports/*.md', '../../../README.md', '../../../PROJECT_PLAN.md', '../../../CURRENT_STATE.md'], {
  query: '?raw',
  import: 'default',
}) as Record<string, () => Promise<string>>;

const ASSETS = import.meta.glob(['../../../docs/**/*.{png,gif,jpg,svg}', '../../../tests/reports/**/*.{png,gif,jpg,svg}', '../../../data/plots/*.{png,gif,jpg}'], {
  query: '?url',
  import: 'default',
  eager: true,
}) as Record<string, string>;

const PREFIX = '../../../';
const strip = (k: string) => k.slice(PREFIX.length);

/** Repo-relative path -> loader. */
export const DOCS: Record<string, () => Promise<string>> = Object.fromEntries(Object.entries(TEXTS).map(([k, v]) => [strip(k), v]));

export function assetUrl(repoPath: string): string | null {
  return ASSETS[PREFIX + repoPath] ?? null;
}

/** Route for a document: #/docs/<repo path without .md>. */
export function docRoute(repoPath: string): string | null {
  return DOCS[repoPath] ? `/docs/${repoPath.replace(/\.md$/, '')}` : null;
}

export function docPathFromRoute(path: string): string {
  const rest = path.replace(/^\/docs\/?/, '');
  return rest ? `${rest}.md` : 'README.md';
}

/** Title: the first Markdown heading. */
export function titleOf(md: string, fallback: string): string {
  const m = md.match(/^#\s+(.+)$/m);
  return m ? m[1].replace(/[`*]/g, '') : fallback;
}

/** A report's display title: its H1 without the trailing date, which is shown separately. */
export function reportTitle(md: string, fallback: string): string {
  return titleOf(md, fallback)
    .replace(/\s*[—–-]\s*\d{4}-\d{2}-\d{2}\s*$/, '')
    .replace(/,?\s*\(?\d{4}-\d{2}-\d{2}\)?\s*$/, '')
    .trim();
}

/** "YYYY-MM-DD" from a file name such as "imaging_2026_09_24.md" (or ""). */
export function dateOfPath(path: string): string {
  const m = path.match(/(\d{4})_(\d{2})_(\d{2})/);
  return m ? `${m[1]}-${m[2]}-${m[3]}` : '';
}

/** Scroll the app's scroll container (the page element, not the window) to the top. */
export function scrollPageTop(): void {
  document.querySelector('.page')?.scrollTo(0, 0);
}
