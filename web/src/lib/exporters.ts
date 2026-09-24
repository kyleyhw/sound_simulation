/** Exports: PNG screenshot, WebM/MP4 video (MediaRecorder), GIF (gifenc). */

import { applyPalette, GIFEncoder, quantize } from 'gifenc';

export function download(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

/** Flatten the visible stack (field canvas + overlays) into one canvas. */
export function composite(wrap: HTMLElement): HTMLCanvasElement {
  const base = wrap.querySelector('canvas') as HTMLCanvasElement;
  const out = document.createElement('canvas');
  out.width = base.width;
  out.height = base.height;
  const ctx = out.getContext('2d')!;
  ctx.drawImage(base, 0, 0);
  for (const c of Array.from(wrap.querySelectorAll('canvas.overlay')) as HTMLCanvasElement[]) {
    if (c.width) ctx.drawImage(c, 0, 0, out.width, out.height);
  }
  const svg = wrap.querySelector('svg.overlay') as SVGSVGElement | null;
  if (svg) {
    // Rasterise markers synchronously is not possible for SVG; the field
    // itself carries the physics, markers are drawn as simple dots.
    const vb = svg.viewBox.baseVal;
    const sx = out.width / vb.width;
    const sy = out.height / vb.height;
    const style = getComputedStyle(document.documentElement);
    for (const g of Array.from(svg.querySelectorAll('g[data-testid]'))) {
      const dot = g.querySelector('circle:last-of-type') as SVGCircleElement | null;
      if (!dot) continue;
      ctx.fillStyle = style.getPropertyValue(g.getAttribute('data-testid') === 'driver-marker' ? '--driver' : '--probe');
      ctx.beginPath();
      ctx.arc(dot.cx.baseVal.value * sx, dot.cy.baseVal.value * sy, Math.max(3, 0.9 * dot.r.baseVal.value * sx), 0, Math.PI * 2);
      ctx.fill();
    }
  }
  return out;
}

export async function screenshot(wrap: HTMLElement, name: string): Promise<void> {
  const c = composite(wrap);
  const blob = await new Promise<Blob | null>((res) => c.toBlob(res, 'image/png'));
  if (blob) download(blob, name);
}

/** Record the field canvas for `ms` milliseconds as WebM (or MP4 on Safari). */
export async function recordVideo(canvas: HTMLCanvasElement, ms: number, name: string): Promise<void> {
  const stream = canvas.captureStream(30);
  const mime = ['video/webm;codecs=vp9', 'video/webm', 'video/mp4'].find((m) => MediaRecorder.isTypeSupported(m));
  if (!mime) throw new Error('video recording is not supported in this browser');
  const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 6_000_000 });
  const chunks: Blob[] = [];
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  const done = new Promise<void>((res) => (rec.onstop = () => res()));
  rec.start(250);
  await new Promise((r) => setTimeout(r, ms));
  rec.stop();
  await done;
  download(new Blob(chunks, { type: mime }), name + (mime.includes('mp4') ? '.mp4' : '.webm'));
}

/** Capture `frames` frames (one per animation frame) into an animated GIF. */
export async function recordGif(wrap: HTMLElement, frames: number, name: string, maxWidth = 480): Promise<void> {
  const gif = GIFEncoder();
  let w = 0;
  let h = 0;
  for (let f = 0; f < frames; f++) {
    await new Promise((r) => requestAnimationFrame(() => r(null)));
    const src = composite(wrap);
    const scale = Math.min(1, maxWidth / src.width);
    w = Math.round(src.width * scale);
    h = Math.round(src.height * scale);
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    const ctx = c.getContext('2d')!;
    ctx.drawImage(src, 0, 0, w, h);
    const { data } = ctx.getImageData(0, 0, w, h);
    const palette = quantize(data, 256);
    gif.writeFrame(applyPalette(data, palette), w, h, { palette, delay: 40 });
  }
  gif.finish();
  download(new Blob([gif.bytes() as BlobPart], { type: 'image/gif' }), name + '.gif');
}
