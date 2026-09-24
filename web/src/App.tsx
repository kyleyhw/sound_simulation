import { Moon, Sun } from 'lucide-react';
import { lazy, Suspense, useEffect } from 'react';
import { navigate, useRoute } from './lib/router';
import { Gallery } from './pages/Gallery';
import { HelpDialog, Sandbox } from './pages/Sandbox';
import { useApp } from './state/store';

const Learn = lazy(() => import('./pages/Learn'));
const Research = lazy(() => import('./pages/Research'));
const Lab = lazy(() => import('./pages/Lab'));
const Loop = lazy(() => import('./pages/Loop'));
const Docs = lazy(() => import('./pages/Docs'));

const NAV: { path: string; label: string }[] = [
  { path: '/', label: 'Sandbox' },
  { path: '/gallery', label: 'Gallery' },
  { path: '/learn', label: 'Learn' },
  { path: '/research', label: 'Research' },
  { path: '/loop', label: 'Loop' },
  { path: '/lab', label: 'Lab' },
  { path: '/docs', label: 'Docs' },
];

function Logo() {
  return (
    <svg width="22" height="22" viewBox="0 0 32 32" aria-hidden>
      <circle cx="16" cy="16" r="4" fill="currentColor" />
      <circle cx="16" cy="16" r="9" fill="none" stroke="currentColor" strokeWidth="2" opacity=".6" />
      <circle cx="16" cy="16" r="14" fill="none" stroke="currentColor" strokeWidth="2" opacity=".3" />
    </svg>
  );
}

function GithubIcon() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <path d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.37-3.88-1.37-.52-1.33-1.28-1.69-1.28-1.69-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.78 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.84 1.19 3.1 0 4.42-2.69 5.39-5.26 5.68.41.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .5Z" />
    </svg>
  );
}

function Toast() {
  const toast = useApp((s) => s.toast);
  if (!toast) return null;
  return (
    <div className={`toast${toast.kind === 'error' ? ' error' : ''}`} role={toast.kind === 'error' ? 'alert' : 'status'} data-testid="toast">
      {toast.text}
    </div>
  );
}

export default function App() {
  const route = useRoute();
  const theme = useApp((s) => s.theme);
  const setTheme = useApp((s) => s.setTheme);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  const base = route.path.split('/').slice(0, 2).join('/') || '/';
  const path = base === '/sandbox' ? '/' : base;
  let page: React.ReactNode;
  switch (path) {
    case '/gallery':
      page = <Gallery />;
      break;
    case '/learn':
      page = <Learn route={route} />;
      break;
    case '/research':
      page = <Research />;
      break;
    case '/lab':
      page = <Lab />;
      break;
    case '/loop':
      page = <Loop />;
      break;
    case '/docs':
      page = <Docs route={route} />;
      break;
    default:
      page = <Sandbox route={route} />;
  }

  return (
    <div className="shell">
      <header className="topbar">
        <a className="brand" href="#/" onClick={(e) => (e.preventDefault(), navigate('/'))}>
          <Logo />
          <span>Acoustic Sandbox</span>
        </a>
        <nav className="nav" aria-label="Main">
          {NAV.map((n) => (
            <a key={n.path} href={`#${n.path}`} aria-current={path === n.path ? 'page' : undefined}>
              {n.label}
            </a>
          ))}
        </nav>
        <span className="spacer" />
        <button
          className="btn icon ghost"
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
          title="Toggle theme"
          data-testid="theme"
        >
          {theme === 'dark' ? <Sun size={17} /> : <Moon size={17} />}
        </button>
        <a className="btn icon ghost" href="https://github.com/kyleyhw/sound_simulation" target="_blank" rel="noreferrer" aria-label="Source code on GitHub">
          <GithubIcon />
        </a>
      </header>
      <main className="page">
        <Suspense fallback={<div className="content muted">Loading…</div>}>{page}</Suspense>
      </main>
      <HelpDialog />
      <Toast />
    </div>
  );
}
