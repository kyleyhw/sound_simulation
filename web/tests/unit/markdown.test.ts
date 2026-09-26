/** Markdown rendering for the docs site: maths, heading ids and in-page anchors. */
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { renderMarkdown, slugify } from '../../src/lib/markdown';

const opts = (path: string, self: string | null = '/docs/docs/x') => ({
  path,
  resolveAsset: () => null,
  routeFor: (p: string) => (p === path ? self : p.endsWith('.md') ? `/docs/${p.replace(/\.md$/, '')}` : null),
});

describe('in-page anchors', () => {
  it('gives headings GitHub-style ids, deduplicated', () => {
    const html = renderMarkdown('# Limitations\n\n## Why *this*, not that?\n\n## Limitations', opts('docs/x.md'));
    expect(html).toContain('<h1 id="limitations">');
    expect(html).toContain('<h2 id="why-this-not-that">');
    expect(html).toContain('<h2 id="limitations-1">');
    expect(slugify('3D  CPML (plan 10.2)')).toBe('3d--cpml-plan-102');
  });

  it('keeps #anchor links away from the hash router', () => {
    const html = renderMarkdown('See [[1]](#ref-cuda-guide) and [Limitations](#limitations).', opts('docs/gpu.md', '/docs/docs/gpu'));
    expect(html).toContain('href="#/docs/docs/gpu?h=ref-cuda-guide" data-anchor="ref-cuda-guide"');
    expect(html).toContain('data-anchor="limitations"');
    // A bare "#ref-cuda-guide" href would be routed to the Sandbox.
    expect(html).not.toContain('href="#ref-cuda-guide"');
  });

  it('keeps cross-document fragments as ?h=', () => {
    const html = renderMarkdown('[physics](physics.md#cpml)', opts('docs/gpu.md'));
    expect(html).toContain('href="#/docs/docs/physics?h=cpml"');
  });
});

describe('maths', () => {
  it('renders inline maths that wraps onto the next source line', () => {
    const html = renderMarkdown('2D $512^2 \\approx 0.11$ ms vs 3D $64^3\n\\approx 0.21$ ms/step.', opts('docs/simulate.md'));
    expect(html).not.toContain('$');
    expect((html.match(/class="katex"/g) ?? []).length).toBe(2);
  });

  it('does not treat currency or a blank-line gap as maths', () => {
    const html = renderMarkdown('It costs $5 and\n$10 in total.\n\nA $x$ here.\n\nPrice $3\n\nthen $4.', opts('docs/x.md'));
    expect((html.match(/class="katex"/g) ?? []).length).toBe(1);
    expect(html).toContain('$5');
    expect(html).toContain('$3');
  });

  it('every published document renders without KaTeX errors', () => {
    const root = join(__dirname, '../../..');
    const files = [
      ...readdirSync(join(root, 'docs'))
        .filter((f) => f.endsWith('.md'))
        .map((f) => `docs/${f}`),
      ...readdirSync(join(root, 'tests/reports'))
        .filter((f) => f.endsWith('.md'))
        .map((f) => `tests/reports/${f}`),
      'README.md',
    ];
    const bad: string[] = [];
    for (const f of files) {
      const html = renderMarkdown(readFileSync(join(root, f), 'utf8'), opts(f));
      if (html.includes('katex-error')) bad.push(`${f}: ${html.match(/katex-error[^>]*title="([^"]*)"/)?.[1] ?? 'katex error'}`);
    }
    expect(bad).toEqual([]);
  });
});
