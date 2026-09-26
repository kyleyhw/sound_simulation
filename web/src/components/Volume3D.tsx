import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { useApp } from '../state/store';

/*
 * Volume renderer for 3D grids: front-to-back alpha compositing along rays
 * through the unit cube (ray-marched in a fragment shader), with a
 * diverging transfer function on the signed pressure and opacity
 * alpha = alphaScale * |p/p_max|^gamma. Opacity is corrected for the step
 * length so the look does not depend on the step count. Wall voxels are
 * drawn in grey. Ported from the previous UI's Volume.tsx.
 *
 * Texture layout: the field index (i*ny + j)*nz + k has k fastest, so the
 * Data3DTexture is (width = nz, height = ny, depth = nx) and box-local
 * (x, y, z) = (k, j, i) / shape.
 */

const VS = /* glsl */ `
precision highp float;
in vec3 position;
uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform vec3 uCameraPos;
out vec3 vOrigin;
out vec3 vDirection;
void main() {
  vOrigin = (inverse(modelMatrix) * vec4(uCameraPos, 1.0)).xyz;
  vDirection = position - vOrigin;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

const FS = /* glsl */ `
precision highp float;
precision highp sampler3D;
in vec3 vOrigin;
in vec3 vDirection;
out vec4 fragColor;
uniform sampler3D uPressure;
uniform sampler3D uWalls;
uniform int uSteps;
uniform float uGamma;
uniform float uAlphaScale;
uniform float uWallAlpha;
uniform vec3 uNeg;
uniform vec3 uPos;
uniform vec3 uWall;
const float REF_STEPS = 128.0;
vec2 hitBox(vec3 o, vec3 d) {
  vec3 inv = 1.0 / d;
  vec3 t0 = (vec3(0.0) - o) * inv;
  vec3 t1 = (vec3(1.0) - o) * inv;
  vec3 tmin = min(t0, t1);
  vec3 tmax = max(t0, t1);
  return vec2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}
void main() {
  vec3 dir = normalize(vDirection);
  vec2 t = hitBox(vOrigin, dir);
  if (t.y <= max(t.x, 0.0)) discard;
  float t0 = max(t.x, 0.0);
  float dt = (t.y - t0) / float(uSteps);
  vec3 p = vOrigin + dir * t0;
  vec4 acc = vec4(0.0);
  for (int i = 0; i < 512; i++) {
    if (i >= uSteps) break;
    float pn = 2.0 * texture(uPressure, p).r - 1.0;
    float w = texture(uWalls, p).r;
    vec3 c;
    float a;
    if (w > 0.5) { c = uWall; a = uWallAlpha; }
    else { c = pn >= 0.0 ? uPos : uNeg; a = uAlphaScale * pow(abs(pn), uGamma); }
    a = 1.0 - pow(1.0 - clamp(a, 0.0, 1.0), dt * REF_STEPS);
    acc.rgb += (1.0 - acc.a) * a * c;
    acc.a += (1.0 - acc.a) * a;
    if (acc.a >= 0.99) break;
    p += dir * dt;
  }
  fragColor = acc;
}`;

export function Volume3D() {
  const runtime = useApp((s) => s.runtime);
  const scene = useApp((s) => s.scene);
  const theme = useApp((s) => s.theme);
  const hostRef = useRef<HTMLDivElement>(null);
  const [gamma, setGamma] = useState(1.6);
  const [alpha, setAlpha] = useState(0.35);
  const [wallAlpha, setWallAlpha] = useState(0.06);
  const stateRef = useRef<{
    renderer: THREE.WebGLRenderer;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
    material: THREE.RawShaderMaterial;
    pTex: THREE.Data3DTexture;
    wTex: THREE.Data3DTexture;
    pBytes: Uint8Array;
    wBytes: Uint8Array;
    markers: THREE.Group;
    raf: number;
    scale: number;
  } | null>(null);

  const [nx, ny, nz] = [scene.params.shape[0], scene.params.shape[1], scene.params.shape[2] ?? 1];

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false, preserveDrawingBuffer: true });
    } catch {
      host.textContent = 'WebGL2 is required for the 3D volume view.';
      return;
    }
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    renderer.setSize(host.clientWidth, host.clientHeight);
    host.appendChild(renderer.domElement);
    const scene3 = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, host.clientWidth / Math.max(1, host.clientHeight), 0.01, 100);
    camera.position.set(1.6, 1.25, 1.9);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0.5, 0.5, 0.5);
    controls.enableDamping = true;
    controls.minDistance = 0.9;
    controls.maxDistance = 6;
    controls.update();

    const pBytes = new Uint8Array(nx * ny * nz).fill(128);
    const wBytes = new Uint8Array(nx * ny * nz);
    const mk = (data: Uint8Array, linear: boolean) => {
      const t = new THREE.Data3DTexture(data, nz, ny, nx);
      t.format = THREE.RedFormat;
      t.type = THREE.UnsignedByteType;
      t.minFilter = t.magFilter = linear ? THREE.LinearFilter : THREE.NearestFilter;
      t.wrapR = t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
      t.unpackAlignment = 1;
      t.needsUpdate = true;
      return t;
    };
    const pTex = mk(pBytes, true);
    const wTex = mk(wBytes, false);
    const material = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: VS,
      fragmentShader: FS,
      side: THREE.BackSide,
      transparent: true,
      depthTest: false,
      uniforms: {
        uPressure: { value: pTex },
        uWalls: { value: wTex },
        uCameraPos: { value: new THREE.Vector3() },
        uSteps: { value: 160 },
        uGamma: { value: gamma },
        uAlphaScale: { value: alpha },
        uWallAlpha: { value: wallAlpha },
        uNeg: { value: new THREE.Color(0x3aa0ff) },
        uPos: { value: new THREE.Color(0xff5a3c) },
        uWall: { value: new THREE.Color(0xb8bdc9) },
      },
    });
    const geo = new THREE.BoxGeometry(1, 1, 1);
    geo.translate(0.5, 0.5, 0.5);
    scene3.add(new THREE.Mesh(geo, material));
    const wire = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: 0x8892a8, transparent: true, opacity: 0.35 }));
    scene3.add(wire);
    const markers = new THREE.Group();
    scene3.add(markers);

    const st = { renderer, camera, controls, material, pTex, wTex, pBytes, wBytes, markers, raf: 0, scale: 1e-9 };
    stateRef.current = st;

    const ro = new ResizeObserver(() => {
      renderer.setSize(host.clientWidth, host.clientHeight);
      camera.aspect = host.clientWidth / Math.max(1, host.clientHeight);
      camera.updateProjectionMatrix();
    });
    ro.observe(host);
    const loop = () => {
      controls.update();
      material.uniforms.uCameraPos.value.copy(camera.position);
      renderer.render(scene3, camera);
      st.raf = requestAnimationFrame(loop);
    };
    st.raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(st.raf);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      material.dispose();
      pTex.dispose();
      wTex.dispose();
      geo.dispose();
      host.removeChild(renderer.domElement);
      stateRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nx, ny, nz]);

  // Upload field + walls each frame.
  useEffect(() => {
    const upload = () => {
      const st = stateRef.current;
      if (!st) return;
      const f = runtime.displayField();
      if (f.length !== st.pBytes.length) return;
      let peak = 0;
      for (let i = 0; i < f.length; i++) peak = Math.max(peak, Math.abs(f[i]));
      st.scale = peak > st.scale ? peak : Math.max(peak, st.scale * 0.96, 1e-9);
      const inv = 1 / st.scale;
      for (let i = 0; i < f.length; i++) {
        const v = Math.max(-1, Math.min(1, f[i] * inv));
        st.pBytes[i] = Math.round(127.5 * (v + 1));
      }
      st.pTex.needsUpdate = true;
      const m = runtime.sim.material;
      for (let i = 0; i < m.length; i++) st.wBytes[i] = m[i] ? 255 : 0;
      st.wTex.needsUpdate = true;
    };
    upload();
    return runtime.onFrame(upload);
  }, [runtime, nx, ny, nz]);

  useEffect(() => {
    const st = stateRef.current;
    if (!st) return;
    st.material.uniforms.uGamma.value = gamma;
    st.material.uniforms.uAlphaScale.value = alpha;
    st.material.uniforms.uWallAlpha.value = wallAlpha;
  }, [gamma, alpha, wallAlpha]);

  useEffect(() => {
    const st = stateRef.current;
    if (!st) return;
    st.renderer.setClearColor(theme === 'dark' ? 0x0b0e14 : 0xeef0f5, 1);
    while (st.markers.children.length) st.markers.remove(st.markers.children[0]);
    const add = (pos: number[], color: number) => {
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.014, 12, 12), new THREE.MeshBasicMaterial({ color }));
      m.position.set((pos[2] + 0.5) / nz, (pos[1] + 0.5) / ny, (pos[0] + 0.5) / nx);
      st.markers.add(m);
    };
    scene.drivers.forEach((d) => add(d.pos, 0xfb7185));
    scene.probes.forEach((p) => add(p.pos, 0xfbbf24));
  }, [scene.drivers, scene.probes, theme, nx, ny, nz]);

  return (
    <div className="viewport" style={{ padding: 0 }}>
      <div ref={hostRef} style={{ position: 'absolute', inset: 0 }} data-testid="volume" />
      <div className="legend volume-controls" style={{ width: 230 }} data-testid="volume-controls">
        <div className="field" style={{ marginBottom: 4 }}>
          <label>
            <span>Opacity</span>
            <span className="mono">{alpha.toFixed(2)}</span>
          </label>
          <input type="range" min={0.02} max={1} step={0.01} value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} aria-label="Opacity" />
        </div>
        <div className="field" style={{ marginBottom: 4 }}>
          <label>
            <span>Contrast (γ)</span>
            <span className="mono">{gamma.toFixed(1)}</span>
          </label>
          <input type="range" min={0.5} max={4} step={0.1} value={gamma} onChange={(e) => setGamma(Number(e.target.value))} aria-label="Contrast" />
        </div>
        <div className="field" style={{ marginBottom: 0 }}>
          <label>
            <span>Wall opacity</span>
            <span className="mono">{wallAlpha.toFixed(2)}</span>
          </label>
          <input type="range" min={0} max={0.5} step={0.01} value={wallAlpha} onChange={(e) => setWallAlpha(Number(e.target.value))} aria-label="Wall opacity" />
        </div>
      </div>
      <div className="hud">Drag to orbit · scroll to zoom · red = compression, blue = rarefaction</div>
    </div>
  );
}
