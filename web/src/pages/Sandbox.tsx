import { X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { Dock } from '../components/Dock';
import { Inspector } from '../components/Inspector';
import { StageToolbar } from '../components/StageToolbar';
import { TOOL_KEYS, ToolRail } from '../components/ToolRail';
import { Viewport } from '../components/Viewport';
import { Volume3D } from '../components/Volume3D';
import { presetById } from '../engine/presets';
import { decodeSceneUrl } from '../engine/scene';
import type { Route } from '../lib/router';
import { fromScene } from '../state/editable';
import { useApp } from '../state/store';
import { markUrlSceneConsumed, urlSceneConsumed } from '../state/urlScene';

function isTyping(e: KeyboardEvent): boolean {
  const t = e.target as HTMLElement | null;
  return !!t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable);
}

/** A focused control that Space activates itself (button, tab, link, summary…). */
function spaceActivates(e: KeyboardEvent): boolean {
  const t = e.target as HTMLElement | null;
  if (!t || typeof t.closest !== 'function') return false;
  return !!t.closest('button, a[href], summary, [role="button"], [role="tab"], [role="menuitem"], [role="radio"], [role="checkbox"], [role="switch"], [role="option"]');
}

export function useSandboxShortcuts(): void {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e)) return;
      const st = useApp.getState();
      // The help dialog is modal: only its own keys work behind it.
      if (st.showHelp && e.key !== '?' && e.key !== 'Escape') return;
      const mod = e.ctrlKey || e.metaKey;
      if (mod && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        if (e.shiftKey) st.redo();
        else st.undo();
        return;
      }
      if (mod && e.key.toLowerCase() === 'y') {
        e.preventDefault();
        st.redo();
        return;
      }
      if (mod) return;
      if (e.key === ' ') {
        // Space on a focused button activates the button, not the simulation.
        if (spaceActivates(e)) return;
        e.preventDefault();
        st.runtime.toggle();
      } else if (e.key === '.') st.runtime.stepOnce(1);
      // With Shift held, e.key is "R", not "r".
      else if (e.key.toLowerCase() === 'r' && e.shiftKey) st.runtime.reset();
      else if (e.key === '?') st.setShowHelp(!st.showHelp);
      else if (e.key === 'Escape') {
        st.setShowHelp(false);
        st.select(null);
      } else if ((e.key === 'Delete' || e.key === 'Backspace') && st.selected) {
        if (st.selected.kind === 'driver') st.removeDriver(st.selected.id);
        else st.removeProbe(st.selected.id);
      } else if (e.key === '[') st.setBrushSize(st.brushSize - 1);
      else if (e.key === ']') st.setBrushSize(st.brushSize + 1);
      else if (TOOL_KEYS[e.key.toLowerCase()] && !e.shiftKey) st.setTool(TOOL_KEYS[e.key.toLowerCase()]);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
}

const FOCUSABLE = 'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

export function HelpDialog() {
  const show = useApp((s) => s.showHelp);
  const setShow = useApp((s) => s.setShowHelp);
  const dialogRef = useRef<HTMLDivElement>(null);
  // Modal focus: take focus on open, keep Tab inside, give it back on close.
  useEffect(() => {
    if (!show) return;
    const back = document.activeElement as HTMLElement | null;
    dialogRef.current?.querySelector<HTMLElement>('[data-autofocus]')?.focus();
    return () => {
      if (back && document.contains(back)) back.focus();
    };
  }, [show]);
  if (!show) return null;
  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      e.stopPropagation();
      setShow(false);
      return;
    }
    if (e.key !== 'Tab' || !dialogRef.current) return;
    const items = Array.from(dialogRef.current.querySelectorAll<HTMLElement>(FOCUSABLE));
    if (!items.length) return;
    const first = items[0];
    const last = items[items.length - 1];
    const active = document.activeElement;
    if (e.shiftKey && (active === first || !dialogRef.current.contains(active))) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && (active === last || !dialogRef.current.contains(active))) {
      e.preventDefault();
      first.focus();
    }
  };
  const rows: [string, string][] = [
    ['Space', 'Run / pause'],
    ['.', 'Single step'],
    ['Shift + R', 'Reset the field'],
    ['V B E L R O', 'Select, brush, eraser, line, rectangle, ellipse'],
    ['C', 'Sound-speed brush (lenses, gradients)'],
    ['S / M', 'Place a source / a microphone'],
    ['Z', 'Draw a control zone (loud / quiet)'],
    ['[ / ]', 'Smaller / larger brush'],
    ['Shift while drawing', 'Filled rectangle or ellipse'],
    ['Delete', 'Remove the selected source or microphone'],
    ['Ctrl + Z / Ctrl + Shift + Z', 'Undo / redo'],
    ['?', 'This help'],
    ['Esc', 'Close this help / deselect'],
  ];
  return (
    <div className="backdrop" onClick={() => setShow(false)} role="presentation">
      <div
        ref={dialogRef}
        className="dialog"
        role="dialog"
        aria-modal="true"
        aria-label="Keyboard shortcuts"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={onKeyDown}
        data-testid="help-dialog"
      >
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <h2>Keyboard shortcuts</h2>
          <button className="btn icon ghost" onClick={() => setShow(false)} aria-label="Close" data-autofocus>
            <X size={16} />
          </button>
        </div>
        <div className="shortcuts">
          {rows.map(([k, v]) => (
            <span key={k} style={{ display: 'contents' }}>
              <span>
                {k.split(' / ').map((part, i) => (
                  <span key={part}>
                    {i > 0 && ' / '}
                    <span className="kbd">{part}</span>
                  </span>
                ))}
              </span>
              <span className="muted">{v}</span>
            </span>
          ))}
        </div>
        <h3 style={{ marginTop: 18 }}>What am I looking at?</h3>
        <p className="muted" style={{ fontSize: 13.5 }}>
          The canvas shows acoustic pressure on a grid, computed with the finite-difference time-domain (FDTD) method: every step updates each
          cell from its neighbours using the wave equation. Walls are painted cells; sources inject a waveform; microphones record the pressure at a point.
        </p>
      </div>
    </div>
  );
}

function FirstRunHint() {
  const [open, setOpen] = useState(() => {
    try {
      return localStorage.getItem('hint-dismissed') !== '1';
    } catch {
      return true;
    }
  });
  if (!open) return null;
  const close = () => {
    setOpen(false);
    try {
      localStorage.setItem('hint-dismissed', '1');
    } catch {
      /* ignore */
    }
  };
  return (
    <div className="hint-card" role="note" data-testid="first-run-hint">
      <p>
        <strong>Welcome.</strong>{' '}
        <span className="hint-wide">
          Press <span className="kbd">Space</span> to run; paint walls, add sources (S) and mics (M).
          <br />
          <span className="dim">
            Or open a ready-made experiment from the <a href="#/gallery">Gallery</a>.
          </span>
        </span>
        <span className="hint-narrow">Tap Run, then paint walls and add sources.</span>
      </p>
      <button className="btn sm ghost hint-wide" onClick={() => useApp.getState().setShowHelp(true)}>
        Shortcuts
      </button>
      <button className="btn sm primary" onClick={close}>
        Got it
      </button>
    </div>
  );
}

export function Sandbox({ route }: { route: Route }) {
  const is3d = useApp((s) => s.scene.params.dims === 3);
  const view3d = useApp((s) => s.view.view3d);
  useSandboxShortcuts();

  // Scene from the URL: ?s=<shared scene> or ?preset=<id>. Loaded once per
  // history entry, so browser Back onto the entry keeps the edits made since.
  useEffect(() => {
    const token = route.query.get('s');
    const presetId = route.query.get('preset');
    const st = useApp.getState();
    const key = route.query.toString();
    if (!token && !presetId) return;
    if (urlSceneConsumed(key)) return;
    markUrlSceneConsumed(key);
    if (token) {
      decodeSceneUrl(token)
        .then((s) => {
          st.loadScene(fromScene(s));
          st.notify(`Loaded shared scene “${s.name}”`);
        })
        .catch((e) => st.notify(`Could not open the shared scene: ${(e as Error).message}`, 'error'));
    } else if (presetId) {
      const p = presetById(presetId);
      if (p) st.loadScene(fromScene(p.build()));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [route.query.toString()]);

  // Stop the loop when leaving the page.
  useEffect(() => () => useApp.getState().runtime.stop(), []);

  return (
    <div className="sandbox">
      <ToolRail />
      <section className="stage" aria-label="Simulation">
        <StageToolbar />
        {is3d && view3d === 'volume' ? <Volume3D /> : <Viewport />}
        <FirstRunHint />
      </section>
      <Dock />
      <Inspector />
    </div>
  );
}
