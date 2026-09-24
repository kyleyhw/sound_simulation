/**
 * The published documents (docs site, write-ups): the repository's Markdown,
 * bundled at build time and loaded on demand per page.
 */

const TEXTS = import.meta.glob(['../../../docs/**/*.md', '../../../tests/reports/*.md', '../../../README.md', '../../../PROJECT_PLAN.md', '../../../CURRENT_STATE.md'], {
  query: '?raw',
  import: 'default',
}) as Record<string, () => Promise<string>>;

const ASSETS = import.meta.glob(['../../../docs/**/*.{png,gif,jpg,svg}', '../../../tests/reports/**/*.{png,gif,jpg,svg}'], {
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
  return rest ? `${rest}.md` : 'docs/index.md';
}

/** Title: the first Markdown heading. */
export function titleOf(md: string, fallback: string): string {
  const m = md.match(/^#\s+(.+)$/m);
  return m ? m[1].replace(/[`*]/g, '') : fallback;
}
