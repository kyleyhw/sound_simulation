import { useLayoutEffect, useMemo, useRef } from 'react';
import { LiveSim, type Variant } from '../components/LiveSim';
import { presetById } from '../engine/presets';
import { encodeRle, type Scene } from '../engine/scene';
import { DEFAULT_PARAMS, type SimParams } from '../engine/simulation';
import { scrollPageToTop } from '../lib/anchors';
import { renderMarkdown } from '../lib/markdown';
import type { Route } from '../lib/router';

/** Interactive explainers (plan 10.6): short articles with live simulations. */

type Block = string | { sim: Variant[]; caption?: string; height?: number };

interface Article {
  id: string;
  title: string;
  blurb: string;
  blocks: Block[];
}

const preset = (id: string, label?: string): Variant => ({ label: label ?? presetById(id)!.title, build: () => presetById(id)!.build(), presetId: id });

function mini(name: string, rows: number, cols: number, params: Partial<SimParams>, fill: (m: Uint8Array) => void, drivers: Scene['drivers'], probes: Scene['probes'] = []): Scene {
  const m = new Uint8Array(rows * cols);
  fill(m);
  return {
    version: 1,
    name,
    units: 'grid',
    params: { ...DEFAULT_PARAMS, ...params, shape: [rows, cols] },
    materials: encodeRle(m),
    drivers,
    probes,
  };
}

const ricker = (f = 0.1, delay = 20) => ({ type: 'ricker' as const, amplitude: 5, frequency: f, delay });

function wall(material: number, label: string): Variant {
  return {
    label,
    build: () =>
      mini(
        `Reflection: ${label}`,
        140,
        220,
        { outer: 'cpml', cpmlCells: 14 },
        (m) => {
          if (material) for (let i = 0; i < 140; i++) for (let j = 150; j < 156; j++) m[i * 220 + j] = material;
        },
        [{ id: 'd', pos: [70, 70], waveform: ricker(0.08, 25), enabled: true }],
        [{ id: 'p', pos: [70, 110], label: 'Listener' }],
      ),
  };
}

const ARTICLES: Article[] = [
  {
    id: 'fdtd',
    title: 'How the simulator works',
    blurb: 'The wave equation on a grid, one step at a time.',
    blocks: [
      `Sound in air is a small pressure disturbance $p(\\mathbf x, t)$ that obeys the **wave equation**

$$\\frac{\\partial^2 p}{\\partial t^2} = c^2 \\nabla^2 p, \\qquad c \\approx 343\\ \\text{m/s}.$$

The simulator stores $p$ on a grid of cells $\\Delta x$ apart, and steps it forward in time by $\\Delta t$ with the *leap-frog* rule

$$p^{n+1} = 2p^n - p^{n-1} + \\Big(\\frac{c\\,\\Delta t}{\\Delta x}\\Big)^2 \\big(p_{\\text{left}} + p_{\\text{right}} + p_{\\text{up}} + p_{\\text{down}} - 4p\\big).$$

Each new value depends only on the cell and its four neighbours, so a wave can move at most one cell per step. The scheme is stable only if $c\\,\\Delta t/\\Delta x \\le 1/\\sqrt{2}$ in 2D (the CFL condition). Below, a single pulse spreads out and bounces off two obstacles. Every run in this app is this rule, repeated.`,
      { sim: [preset('pulse-room')], caption: 'A Ricker pulse in a room with pressure-release walls (p = 0): every reflection flips the sign.' },
      `**How accurate is it?** The scheme is second-order accurate: halving $\\Delta x$ quarters the error. It conserves energy to one part in $10^7$, and it reproduces the analytic room resonances to better than 0.2 %. The waves it carries travel slightly too slowly at short wavelengths. At 10 cells per wavelength the speed error is 1.5 %. The numbers are in [the physics report](#/docs/docs/physics).`,
    ],
  },
  {
    id: 'walls',
    title: 'Walls: soft, hard and absorbing',
    blurb: 'Why an echo can come back inverted, or not at all.',
    blocks: [
      `When a wave meets a wall, part of it reflects. For a wall with specific impedance $Z$ at normal incidence, the reflection coefficient is

$$R = \\frac{Z - \\rho c}{Z + \\rho c}.$$

- A **pressure-release** surface ($Z \\to 0$, like the edge of the water in a pipe) gives $R = -1$: the echo comes back inverted.
- A **rigid** wall ($Z \\to \\infty$) gives $R = +1$: the echo comes back unchanged.
- An **absorber** matched to air ($Z = \\rho c$) gives $R = 0$ at normal incidence. It still reflects at grazing angles.

A **perfectly matched layer** (PML) is a numerical trick. It stretches space into the complex plane, so waves enter it at any angle and decay without reflecting. The simulator's PML reflects less than −49 dB even at 62°. Try each wall below and watch what comes back towards the source.`,
      {
        sim: [wall(1, 'Soft (p = 0)'), wall(2, 'Rigid'), wall(5, 'Absorber'), wall(0, 'No wall (PML)')],
        caption: 'The same pulse and four walls. The outer edges are a PML, so the only echo is from the wall.',
        height: 260,
      },
    ],
  },
  {
    id: 'rooms',
    title: 'Room modes and reverberation',
    blurb: 'Why a room hums, and how long it rings.',
    blocks: [
      `A closed room only supports certain standing waves, its **modes**. For a rectangular room of size $L_x \\times L_y$ with rigid walls, they are

$$f_{mn} = \\frac{c}{2}\\sqrt{\\Big(\\frac{m}{L_x}\\Big)^2 + \\Big(\\frac{n}{L_y}\\Big)^2}, \\qquad m, n = 0, 1, 2, \\dots$$

Drive a room at one of these frequencies and a fixed pattern of loud and silent lines appears.`,
      { sim: [preset('room-modes')], caption: 'A source at a room resonance builds up a standing-wave pattern.' },
      `Real walls absorb part of every reflection, so sound decays. The **reverberation time** $T_{60}$ is the time for the energy to drop by 60 dB. Sabine's formula links it to the room volume $V$, surface area $S$ and mean absorption $\\alpha$:

$$T_{60} \\approx \\frac{24 \\ln 10}{c}\\,\\frac{V}{S\\,\\alpha}.$$

In the simulator's own tests, the measured $T_{60}$ of rooms with absorbing walls falls between the Sabine and Eyring predictions. The [Lab](#/lab) measures the $T_{60}$ of the room you are sitting in.`,
    ],
  },
  {
    id: 'echolocation',
    title: 'Echolocation: hearing a room',
    blurb: 'Echo delays are distances. How far does that get a laptop?',
    blocks: [
      `A click from a speaker and its echo off a wall $d$ metres away are separated by the round trip

$$\\tau = \\frac{2d}{c},$$

which is 5.8 ms per metre. With a pair of microphones, the tiny difference in arrival times between them also gives a direction.`,
      { sim: [preset('echolocation')], caption: 'A short tone burst and its echoes from a wall and a pillar. The yellow microphone sits next to the source.' },
      `**Back-projection.** For every candidate point $\\mathbf x$, add up the recordings at the delay a reflector at $\\mathbf x$ would produce:

$$I(\\mathbf x) = \\sum_{s,m} r_{sm}\\!\\Big(\\frac{|\\mathbf x - \\mathbf x_s| + |\\mathbf x - \\mathbf x_m|}{c}\\Big).$$

Real reflectors add up in phase and everything else cancels. The [closed-loop demo](#/loop) uses exactly this to build a digital twin of the room.

**An honest result.** Our first machine-learning models predicted obstacle maps from stereo recordings with an IoU of 0.10. A baseline that never listens scores 0.10 too: it predicts where obstacles usually are. So those models had learned the room statistics, not the echoes. The [debug audit](#/docs/tests/reports/debug_audit_2026_09_24) and the [plan audit](#/docs/docs/plan_audit) explain how this was found. The second attempt measures itself against that baseline from the start.`,
    ],
  },
  {
    id: 'control',
    title: 'Beams, quiet zones and time reversal',
    blurb: 'Many speakers, one delay each: shaping where sound goes.',
    blocks: [
      `A row of speakers driven with delays that grow along the array, $d_k = k\\,s\\sin\\theta / c$, tilts the combined wavefront by $\\theta$. The beam turns without anything moving.`,
      { sim: [preset('beam-steering'), preset('quiet-zone'), preset('time-reversal')], caption: 'Beam steering, a designed quiet zone, and time-reversal focusing.' },
      `**Quiet zones.** To make one region loud and another silent, measure how each speaker reaches each region: transfer functions $H_b$ (bright) and $H_d$ (dark). Then choose complex weights $\\mathbf w$ (a gain and a delay per speaker) that maximise the ratio of the energies:

$$\\max_{\\mathbf w}\\ \\frac{\\mathbf w^H H_b^H H_b\\,\\mathbf w}{\\mathbf w^H (H_d^H H_d + \\delta I)\\,\\mathbf w}.$$

This is *acoustic contrast control*. The answer is the top generalised eigenvector. In the quiet-zone scene the ten speakers reach 58 dB of contrast. Draw your own zones in the sandbox's **Control** tab.

**Time reversal.** The wave equation looks the same with time running backwards. So record a pulse from a point, then play every recording backwards from where it was recorded. The waves retrace their paths through all the scattering and converge on the original point.`,
    ],
  },
  {
    id: 'diffraction',
    title: 'Diffraction, lenses and whispering galleries',
    blurb: 'Waves bend around corners, focus through gradients and cling to curved walls.',
    blocks: [
      `Waves spread through openings comparable to their wavelength. Two slits give interference fringes at angles $\\sin\\theta = m\\lambda/a$. A region where sound travels more slowly bends rays towards it, which makes a lens with no surfaces at all. Along a concave rigid wall, rays graze and reflect again and again, so a whisper travels around the wall of St Paul's dome.`,
      { sim: [preset('double-slit'), preset('lens'), preset('whispering-gallery')], caption: 'Double-slit interference, a gradient-index lens, and a whispering gallery.' },
    ],
  },
];

export default function Learn({ route }: { route?: Route }) {
  const id = route?.path.split('/')[2];
  const article = ARTICLES.find((a) => a.id === id);
  const rootRef = useRef<HTMLDivElement>(null);
  // Next/Previous keep the same scroller (.page), so start each article at the top.
  useLayoutEffect(() => scrollPageToTop(rootRef.current), [id]);
  const rendered = useMemo(
    () =>
      article?.blocks.map((b) =>
        typeof b === 'string' ? renderMarkdown(b, { path: `learn/${article.id}.md`, resolveAsset: () => null, routeFor: () => null }) : null,
      ),
    [article],
  );

  if (!article) {
    return (
      <div className="content" data-testid="learn" ref={rootRef}>
        <h1>Learn</h1>
        <p className="lede">Short explainers. Each one has a live simulation you can run, pause, switch and open in the sandbox.</p>
        <div className="card-grid">
          {ARTICLES.map((a, k) => (
            <a key={a.id} className="card" href={`#/learn/${a.id}`} data-testid={`article-${a.id}`}>
              <div className="muted mono" style={{ fontSize: 12 }}>
                {k + 1}
              </div>
              <h3 style={{ margin: '4px 0' }}>{a.title}</h3>
              <p className="muted" style={{ margin: 0 }}>
                {a.blurb}
              </p>
            </a>
          ))}
        </div>
      </div>
    );
  }

  const k = ARTICLES.indexOf(article);
  return (
    <div className="content prose" data-testid="learn-article" ref={rootRef}>
      <p className="muted" style={{ fontSize: 13 }}>
        <a href="#/learn">Learn</a> · {k + 1} of {ARTICLES.length}
      </p>
      <h1>{article.title}</h1>
      {article.blocks.map((b, i) =>
        typeof b === 'string' ? (
          <div key={i} dangerouslySetInnerHTML={{ __html: rendered![i]! }} />
        ) : (
          <LiveSim key={`${article.id}-${i}`} variants={b.sim} caption={b.caption} height={b.height} />
        ),
      )}
      <div className="btn-row" style={{ marginTop: 28 }}>
        {k > 0 && (
          <a className="btn" href={`#/learn/${ARTICLES[k - 1].id}`}>
            ← {ARTICLES[k - 1].title}
          </a>
        )}
        {k < ARTICLES.length - 1 && (
          <a className="btn primary" href={`#/learn/${ARTICLES[k + 1].id}`}>
            {ARTICLES[k + 1].title} →
          </a>
        )}
      </div>
    </div>
  );
}
