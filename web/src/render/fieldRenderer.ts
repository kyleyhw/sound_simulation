/**
 * 2D field renderer. WebGL2 path: the float field is uploaded as an R32F
 * texture at native grid resolution and mapped through a colormap LUT in
 * the fragment shader (nearest sampling = crisp cells). Materials are a
 * second R8 texture composited on top. Falls back to Canvas2D ImageData
 * when WebGL2 is unavailable.
 */

import { colormapLut, type ColormapName } from './colormaps';

export type ScaleMode = 'linear' | 'db';

export interface RenderOptions {
  colormap: ColormapName;
  mode: ScaleMode;
  /** Values map to [-scale, +scale] (linear) or [scale*10^(-range/20), scale] (dB). */
  scale: number;
  dbRange: number;
  /** Signed field (diverging) vs magnitude field (sequential, e.g. RMS). */
  signed: boolean;
  showMaterials: boolean;
  theme: 'dark' | 'light';
}

const VS = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = vec2(a_pos.x * 0.5 + 0.5, 0.5 - a_pos.y * 0.5);
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const FS = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_field;
uniform sampler2D u_lut;
uniform sampler2D u_mat;
uniform float u_scale;
uniform float u_dbRange;
uniform int u_mode;     // 0 linear, 1 dB
uniform int u_signed;   // 1 signed (diverging) else magnitude
uniform int u_showMat;
uniform vec3 u_matColors[6];
out vec4 outColor;
void main() {
  float v = texture(u_field, v_uv).r;
  float t;
  if (u_mode == 1) {
    float mag = abs(v) / max(u_scale, 1e-20);
    float db = 20.0 * log(max(mag, 1e-12)) / log(10.0);
    float x = clamp(1.0 + db / u_dbRange, 0.0, 1.0);
    t = (u_signed == 1) ? 0.5 + 0.5 * sign(v) * x : x;
  } else {
    t = (u_signed == 1) ? clamp(0.5 + 0.5 * v / max(u_scale, 1e-20), 0.0, 1.0)
                        : clamp(v / max(u_scale, 1e-20), 0.0, 1.0);
  }
  vec4 col = texture(u_lut, vec2(t, 0.5));
  if (u_showMat == 1) {
    float m = texture(u_mat, v_uv).r * 255.0;
    int id = int(m + 0.5);
    if (id > 0) col = vec4(u_matColors[min(id, 5)], 1.0);
  }
  outColor = col;
}`;

export const MATERIAL_RGB: Record<'dark' | 'light', [number, number, number][]> = {
  dark: [
    [0, 0, 0],
    [0.55, 0.58, 0.66], // soft
    [0.86, 0.87, 0.9], // rigid
    [0.78, 0.7, 0.55], // plaster
    [0.62, 0.45, 0.3], // wood
    [0.35, 0.55, 0.42], // absorber
  ],
  light: [
    [0, 0, 0],
    [0.45, 0.47, 0.55],
    [0.2, 0.22, 0.27],
    [0.66, 0.58, 0.42],
    [0.52, 0.36, 0.22],
    [0.25, 0.45, 0.33],
  ],
};

export class FieldRenderer {
  private gl: WebGL2RenderingContext | null = null;
  private ctx2d: CanvasRenderingContext2D | null = null;
  private prog: WebGLProgram | null = null;
  private texField: WebGLTexture | null = null;
  private texLut: WebGLTexture | null = null;
  private texMat: WebGLTexture | null = null;
  private lutName: ColormapName | null = null;
  private w = 0;
  private h = 0;
  private image: ImageData | null = null;
  private off: HTMLCanvasElement | null = null;
  readonly backend: 'webgl2' | 'canvas2d';

  constructor(private readonly canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', { antialias: false, preserveDrawingBuffer: true });
    if (gl) {
      this.gl = gl;
      this.backend = 'webgl2';
      this.initGl(gl);
    } else {
      this.ctx2d = canvas.getContext('2d');
      this.backend = 'canvas2d';
    }
  }

  private initGl(gl: WebGL2RenderingContext): void {
    const compile = (type: number, src: string): WebGLShader => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? 'shader error');
      return s;
    };
    const prog = gl.createProgram()!;
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog) ?? 'link error');
    this.prog = prog;
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, 'a_pos');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    const mk = (): WebGLTexture => {
      const t = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return t;
    };
    this.texField = mk();
    this.texLut = mk();
    gl.bindTexture(gl.TEXTURE_2D, this.texLut);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    this.texMat = mk();
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  }

  /**
   * Draw a field of `rows x cols` (row-major, row 0 at the top).
   * `materials` is the matching per-cell material id map (or null).
   */
  draw(field: Float32Array, rows: number, cols: number, materials: Uint8Array | null, opt: RenderOptions): void {
    if (this.gl) this.drawGl(field, rows, cols, materials, opt);
    else this.draw2d(field, rows, cols, materials, opt);
  }

  private drawGl(field: Float32Array, rows: number, cols: number, materials: Uint8Array | null, opt: RenderOptions): void {
    const gl = this.gl!;
    const { width, height } = this.canvas;
    gl.viewport(0, 0, width, height);
    gl.useProgram(this.prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.texField);
    if (rows !== this.h || cols !== this.w) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, cols, rows, 0, gl.RED, gl.FLOAT, field);
      this.w = cols;
      this.h = rows;
    } else {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, cols, rows, gl.RED, gl.FLOAT, field);
    }
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.texLut);
    if (this.lutName !== opt.colormap) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, colormapLut(opt.colormap));
      this.lutName = opt.colormap;
    }
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.texMat);
    const mat = materials ?? new Uint8Array(rows * cols);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, cols, rows, 0, gl.RED, gl.UNSIGNED_BYTE, mat);
    const u = (n: string) => gl.getUniformLocation(this.prog!, n);
    gl.uniform1i(u('u_field'), 0);
    gl.uniform1i(u('u_lut'), 1);
    gl.uniform1i(u('u_mat'), 2);
    gl.uniform1f(u('u_scale'), opt.scale);
    gl.uniform1f(u('u_dbRange'), opt.dbRange);
    gl.uniform1i(u('u_mode'), opt.mode === 'db' ? 1 : 0);
    gl.uniform1i(u('u_signed'), opt.signed ? 1 : 0);
    gl.uniform1i(u('u_showMat'), opt.showMaterials ? 1 : 0);
    gl.uniform3fv(u('u_matColors'), MATERIAL_RGB[opt.theme].flat());
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  private draw2d(field: Float32Array, rows: number, cols: number, materials: Uint8Array | null, opt: RenderOptions): void {
    const ctx = this.ctx2d;
    if (!ctx) return;
    if (!this.image || this.image.width !== cols || this.image.height !== rows) {
      this.image = new ImageData(cols, rows);
      this.off = document.createElement('canvas');
      this.off.width = cols;
      this.off.height = rows;
    }
    const lut = colormapLut(opt.colormap);
    const px = this.image.data;
    const matRgb = MATERIAL_RGB[opt.theme];
    for (let i = 0; i < rows * cols; i++) {
      const m = materials ? materials[i] : 0;
      if (opt.showMaterials && m > 0) {
        const c = matRgb[Math.min(m, 5)];
        px[i * 4] = c[0] * 255;
        px[i * 4 + 1] = c[1] * 255;
        px[i * 4 + 2] = c[2] * 255;
        px[i * 4 + 3] = 255;
        continue;
      }
      const v = field[i];
      let t: number;
      if (opt.mode === 'db') {
        const db = 20 * Math.log10(Math.max(Math.abs(v) / opt.scale, 1e-12));
        const x = Math.min(1, Math.max(0, 1 + db / opt.dbRange));
        t = opt.signed ? 0.5 + 0.5 * Math.sign(v) * x : x;
      } else {
        t = opt.signed ? 0.5 + (0.5 * v) / opt.scale : v / opt.scale;
      }
      const k = Math.round(Math.min(1, Math.max(0, t)) * 255) * 4;
      px[i * 4] = lut[k];
      px[i * 4 + 1] = lut[k + 1];
      px[i * 4 + 2] = lut[k + 2];
      px[i * 4 + 3] = 255;
    }
    const off = this.off!;
    off.getContext('2d')!.putImageData(this.image, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, this.canvas.width, this.canvas.height);
  }
}
