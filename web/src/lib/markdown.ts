/**
 * Markdown with KaTeX maths, for the docs site and write-ups (plan 10.9).
 * Inputs are the repository's own Markdown files (trusted content).
 *
 * Maths ($$...$$ blocks, $...$ inline, and ```math fences) is cut out
 * before Markdown parsing, so the parser cannot mangle underscores and
 * backslashes, and then pasted back as KaTeX HTML. Relative links to other
 * .md files become in-app routes. Relative image paths resolve through
 * `resolveAsset`.
 */
import katex from 'katex';
import 'katex/dist/katex.min.css';
import { marked } from 'marked';

export interface RenderOptions {
  /** Repo-relative path of the document (e.g. "docs/physics.md"). */
  path: string;
  /** Map a repo-relative asset path to a URL (or null to leave it). */
  resolveAsset: (repoPath: string) => string | null;
  /** Route for a repo-relative Markdown path, or null if it is not published. */
  routeFor: (repoPath: string) => string | null;
}

function joinPath(base: string, rel: string): string {
  const parts = base.split('/').slice(0, -1);
  for (const seg of rel.split('/')) {
    if (seg === '..') parts.pop();
    else if (seg !== '.' && seg !== '') parts.push(seg);
  }
  return parts.join('/');
}

export function renderMarkdown(src: string, opts: RenderOptions): string {
  const math: string[] = [];
  const stash = (tex: string, display: boolean) => {
    let html: string;
    try {
      html = katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: 'ignore' });
    } catch {
      html = `<code>${tex}</code>`;
    }
    math.push(html);
    return `\u0000M${math.length - 1}\u0000`;
  };
  // Protect code first so $ inside code is not treated as maths.
  const code: string[] = [];
  let s = src.replace(/```math\n([\s\S]*?)```/g, (_m, tex) => stash(tex, true));
  s = s.replace(/```[\s\S]*?```/g, (m) => {
    code.push(m);
    return `\u0000C${code.length - 1}\u0000`;
  });
  s = s.replace(/`[^`\n]+`/g, (m) => {
    code.push(m);
    return `\u0000C${code.length - 1}\u0000`;
  });
  s = s.replace(/\$\$([\s\S]+?)\$\$/g, (_m, tex) => stash(tex, true));
  s = s.replace(/(^|[^\\$])\$([^$\n]+?)\$(?!\d)/g, (_m, pre, tex) => pre + stash(tex, false));
  s = s.replace(/\u0000C(\d+)\u0000/g, (_m, k) => code[Number(k)]);

  const renderer = new marked.Renderer();
  const baseLink = renderer.link.bind(renderer);
  renderer.link = (tok) => {
    const href = tok.href ?? '';
    if (!/^[a-z]+:|^#|^\//i.test(href)) {
      const [file, hash] = href.split('#');
      const target = joinPath(opts.path, file);
      if (file.endsWith('.md')) {
        const route = opts.routeFor(target);
        if (route) return `<a href="#${route}${hash ? `?h=${encodeURIComponent(hash)}` : ''}">${marked.parseInline(tok.text) as string}</a>`;
      }
      return `<a href="https://github.com/kyleyhw/sound_simulation/blob/main/${target}" target="_blank" rel="noreferrer">${marked.parseInline(tok.text) as string}</a>`;
    }
    return baseLink(tok);
  };
  const baseImage = renderer.image.bind(renderer);
  renderer.image = (tok) => {
    const href = tok.href ?? '';
    if (!/^[a-z]+:|^\//i.test(href)) {
      const url = opts.resolveAsset(joinPath(opts.path, href));
      if (url) return `<img src="${url}" alt="${tok.text ?? ''}" loading="lazy">`;
    }
    return baseImage(tok);
  };
  let html = marked.parse(s, { renderer, gfm: true, async: false }) as string;
  html = html.replace(/\u0000M(\d+)\u0000/g, (_m, k) => math[Number(k)]);
  // Raw HTML <img src="relative"> (e.g. the README hero) resolves like Markdown images.
  html = html.replace(/<img([^>]*?)\ssrc="(?![a-z]+:|\/)([^"]+)"/gi, (m, pre, src) => {
    const url = opts.resolveAsset(joinPath(opts.path, src));
    return url ? `<img${pre} src="${url}"` : m;
  });
  return html;
}
