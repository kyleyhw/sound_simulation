import { ChevronDown, Moon, Sun } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import type { PageId } from '../lib/router';
import { useApp } from '../state/store';

export const PRIMARY: { id: PageId; path: string; label: string }[] = [
  { id: 'echo', path: '/echo', label: 'Echo vision' },
  { id: 'sandbox', path: '/sandbox', label: 'Sandbox' },
  { id: 'gallery', path: '/gallery', label: 'Gallery' },
  { id: 'learn', path: '/learn', label: 'Learn' },
];

export const SECONDARY: { id: PageId; path: string; label: string; blurb: string }[] = [
  { id: 'research', path: '/research', label: 'Research', blurb: 'What we found, negative results included' },
  { id: 'loop', path: '/loop', label: 'Loop', blurb: 'Sense, rebuild and control a changing room' },
  { id: 'lab', path: '/lab', label: 'Lab', blurb: "Measure a real room with your laptop's speakers" },
  { id: 'docs', path: '/docs', label: 'Docs', blurb: 'Physics, code and reports' },
];

export const GITHUB_URL = 'https://github.com/kyleyhw/sound_simulation';

/**
 * Main navigation: four primary sections plus a "More" menu. The menu is a
 * disclosure button with a menu of links: arrow keys move, Escape closes and
 * returns focus, and it closes on an outside click or a route change. On
 * phones the theme toggle and the GitHub link move into the menu so the bar
 * fits in 390 px with nothing hidden.
 */
export function TopNav({ page }: { page: PageId }) {
  const theme = useApp((s) => s.theme);
  const setTheme = useApp((s) => s.setTheme);
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const inMore = SECONDARY.some((s) => s.id === page);

  // Close on a route change.
  useEffect(() => setOpen(false), [page]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('pointerdown', onDown);
    return () => document.removeEventListener('pointerdown', onDown);
  }, [open]);

  const items = () => Array.from(menuRef.current?.querySelectorAll<HTMLElement>('[role="menuitem"]') ?? []).filter((el) => el.offsetParent !== null);

  const focusItem = (k: number) => {
    const all = items();
    if (all.length) all[(k + all.length) % all.length].focus();
  };

  const onButtonKey = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      setOpen(true);
      const last = e.key === 'ArrowUp';
      requestAnimationFrame(() => focusItem(last ? -1 : 0));
    }
  };

  const onMenuKey = (e: React.KeyboardEvent) => {
    const all = items();
    const k = all.indexOf(document.activeElement as HTMLElement);
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      focusItem(k + 1);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      focusItem(k - 1);
    } else if (e.key === 'Home') {
      e.preventDefault();
      focusItem(0);
    } else if (e.key === 'End') {
      e.preventDefault();
      focusItem(-1);
    } else if (e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      setOpen(false);
      buttonRef.current?.focus();
    } else if (e.key === 'Tab') {
      setOpen(false);
    }
  };

  const toggleTheme = () => setTheme(theme === 'dark' ? 'light' : 'dark');

  return (
    <>
      <nav className="nav" aria-label="Main">
        {PRIMARY.map((n) => (
          <a key={n.path} href={`#${n.path}`} aria-current={page === n.id ? 'page' : undefined}>
            {n.label}
          </a>
        ))}
        <div className="more" ref={wrapRef}>
          <button
            ref={buttonRef}
            className="more-btn"
            aria-haspopup="menu"
            aria-expanded={open}
            aria-controls="more-menu"
            data-current={inMore || undefined}
            onClick={() => setOpen((o) => !o)}
            onKeyDown={onButtonKey}
            data-testid="nav-more"
          >
            More <ChevronDown size={14} aria-hidden />
          </button>
          {open && (
            <div className="more-menu" id="more-menu" role="menu" aria-label="More sections" ref={menuRef} onKeyDown={onMenuKey}>
              {SECONDARY.map((n) => (
                <a
                  key={n.path}
                  role="menuitem"
                  href={`#${n.path}`}
                  aria-current={page === n.id ? 'page' : undefined}
                  onClick={() => setOpen(false)}
                >
                  {n.label}
                </a>
              ))}
              <div className="more-extra" role="none">
                <hr role="separator" />
                <button role="menuitem" onClick={() => (toggleTheme(), setOpen(false))}>
                  {theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />} {theme === 'dark' ? 'Light theme' : 'Dark theme'}
                </button>
                <a role="menuitem" href={GITHUB_URL} target="_blank" rel="noreferrer">
                  Source on GitHub
                </a>
              </div>
            </div>
          )}
        </div>
      </nav>
    </>
  );
}
