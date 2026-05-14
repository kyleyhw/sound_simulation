import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// =====================================================================
// Public types — re-declared here so this component stays decoupled from
// App.tsx's wire-payload typing. Caller is responsible for forwarding the
// already-decoded ArrayBuffers and shapes from the socket.
// =====================================================================

/**
 * Buffer-and-shape pair held by App.tsx in a useRef. We pass the ref
 * itself rather than the Uint8Array because React 19's dev-mode
 * profiler tries to deep-clone every prop into the Performance API
 * via `performance.measure`, and a typed array of even tens of KB
 * triggers a DataCloneError that derails the render cycle. Refs are
 * opaque to the profiler, so the heavy buffer never enters that path.
 */
export interface VolumeBuffer {
  bytes: Uint8Array | null;
  shape: [number, number, number] | null;
}

export interface VolumeProps {
  /** Latest pressure buffer, accessed by the upload effect via .current. */
  pressureRef: React.RefObject<VolumeBuffer>;
  /** Monotonic counter bumped each time pressureRef.current is replaced. */
  pressureFrame: number;
  /** Per-frame peak amplitude — for the on-screen readout, not the shader. */
  maxVal: number;

  /** Same pattern for the obstacle mask. */
  obstacleRef: React.RefObject<VolumeBuffer>;
  obstacleFrame: number;

  /**
   * Drivers in *full-grid* coordinates (not the downsampled shape). The
   * fullGridShape prop tells the component what to divide by to land in
   * the volume's [0, 1]^3 box-local frame.
   */
  drivers: { position: [number, number, number] }[];
  fullGridShape: [number, number, number] | null;

  /** Transfer-function controls. */
  gamma: number;
  alphaScale: number;
  obstacleAlpha: number;
  raySteps: number;

  /** Imperative reset signal — bump to reset the orbit camera to the default pose. */
  resetCameraNonce: number;

  /** CSS-pixel size of the canvas. */
  width: number;
  height: number;
}

// =====================================================================
// GLSL — kept as plain string literals rather than separate files so the
// build pipeline (vite) does not need a glsl-loader plugin.
// =====================================================================

// The volume cube's local-space coordinates are [0, 1]^3, which align
// exactly with the texture's UVW. The vertex shader passes the per-fragment
// box-local position to the fragment shader, plus the camera's box-local
// position so the fragment can build the ray.
const vertexShader = /* glsl */ `
  precision highp float;
  in vec3 position;

  uniform mat4 modelMatrix;
  uniform mat4 modelViewMatrix;
  uniform mat4 projectionMatrix;
  uniform vec3 uCameraPos;        // world-space camera, supplied each frame

  out vec3 vOrigin;               // ray origin in box-local coords
  out vec3 vDirection;            // ray direction in box-local coords (un-normalised)

  void main() {
    // Camera position in box-local space.
    vOrigin = (inverse(modelMatrix) * vec4(uCameraPos, 1.0)).xyz;
    // Direction = (this fragment's box-local position) - (camera box-local).
    vDirection = position - vOrigin;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

// Front-to-back alpha-compositing volume rendering.
//
//   C_i += (1 - A_i) * alpha_i * c_i
//   A_i +=  (1 - A_i) * alpha_i
//
// with early termination when A_i >= 0.99. The transfer function is
// diverging blue / white / red on the signed pressure p_norm = 2 * b - 1,
// with per-sample alpha = alphaScale * |p_norm|^gamma. Obstacle voxels
// override colour and alpha. A constant-step ray march keeps the cost
// bounded by raySteps independent of camera position.
const fragmentShader = /* glsl */ `
  precision highp float;
  precision highp sampler3D;

  in vec3 vOrigin;
  in vec3 vDirection;
  out vec4 fragColor;

  uniform sampler3D uPressure;
  uniform sampler3D uObstacle;
  uniform int   uSteps;
  uniform float uGamma;
  uniform float uAlphaScale;
  uniform float uObstacleAlpha;

  // Reference dt at which the user-set alphaScale gives "the right look".
  // Opacity correction below scales the per-step alpha so the visual is
  // step-count-invariant: a sample marched in 64 steps deposits the same
  // total opacity as in 256 steps.
  const float REF_STEPS = 128.0;

  // AABB slab-test: returns (t_near, t_far) for the unit cube [0, 1]^3.
  vec2 intersectBox(vec3 o, vec3 d) {
    vec3 invD = 1.0 / d;
    vec3 t0 = (vec3(0.0) - o) * invD;
    vec3 t1 = (vec3(1.0) - o) * invD;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    float tNear = max(max(tmin.x, tmin.y), tmin.z);
    float tFar  = min(min(tmax.x, tmax.y), tmax.z);
    return vec2(tNear, tFar);
  }

  // Diverging colormap matching the 2D canvas renderer:
  //   v = -1 -> blue (0.16, 0.31, 0.78)
  //   v =  0 -> near-white (0.96)
  //   v = +1 -> red (0.86, 0.16, 0.16)
  vec3 colormap(float v) {
    vec3 zero = vec3(0.96, 0.96, 0.96);
    if (v >= 0.0) return mix(zero, vec3(0.86, 0.16, 0.16), v);
    return mix(zero, vec3(0.16, 0.31, 0.78), -v);
  }

  void main() {
    vec3 dir = normalize(vDirection);
    vec2 t = intersectBox(vOrigin, dir);
    if (t.y <= max(t.x, 0.0)) discard;     // ray missed the cube

    float tStart = max(t.x, 0.0);
    float dt = (t.y - tStart) / float(uSteps);
    vec3 step = dir * dt;
    vec3 p = vOrigin + dir * tStart;

    vec4 acc = vec4(0.0);
    // Loop bound is a hard upper limit so WebGL2 can statically unroll;
    // the runtime cap is uSteps.
    for (int i = 0; i < 512; i++) {
      if (i >= uSteps) break;

      // Sample the pressure texture. R8 is hardware-normalised to [0, 1];
      // unmap to signed pressure in [-1, 1].
      float b   = texture(uPressure, p).r;
      float pn  = 2.0 * b - 1.0;
      float obs = texture(uObstacle, p).r;

      vec3 c;
      float a;
      if (obs > 0.5) {
        // Obstacle voxel: gray, user-controlled opacity.
        c = vec3(0.31, 0.31, 0.35);
        a = uObstacleAlpha;
      } else {
        c = colormap(pn);
        a = uAlphaScale * pow(abs(pn), uGamma);
      }

      // Opacity correction so visual is invariant to step count: the
      // single-sample contribution is converted to "what it would be if
      // the ray were sampled REF_STEPS times across the unit cube".
      a = 1.0 - pow(1.0 - clamp(a, 0.0, 1.0), dt * REF_STEPS);

      acc.rgb += (1.0 - acc.a) * a * c;
      acc.a   += (1.0 - acc.a) * a;

      if (acc.a >= 0.99) break;
      p += step;
    }
    fragColor = acc;
  }
`;

// =====================================================================
// React wrapper
// =====================================================================

const Volume: React.FC<VolumeProps> = (props) => {
  const mountRef = useRef<HTMLDivElement>(null);

  // Three.js objects. Stored in a ref so the React render cycle does not
  // tear them down on every prop change.
  const sceneRef = useRef<{
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
    pressureTex: THREE.Data3DTexture | null;
    obstacleTex: THREE.Data3DTexture | null;
    material: THREE.RawShaderMaterial;
    cube: THREE.Mesh;
    driverGroup: THREE.Group;
    rafHandle: number;
    pressureShapeKey: string; // tracks shape for texture re-allocation
    obstacleShapeKey: string;
  } | null>(null);

  // -- One-time scene setup --------------------------------------------
  useEffect(() => {
    if (!mountRef.current) return;
    const mount = mountRef.current;

    const renderer = new THREE.WebGLRenderer({
      antialias: false,    // a volume renderer's AA comes from filtering, not MSAA
      alpha: false,
    });
    renderer.setPixelRatio(1);  // CSS-resolution rendering per spec
    renderer.setSize(props.width, props.height);
    renderer.setClearColor(0x14171c, 1.0);
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();

    // Camera: 45 deg FOV, looking at the cube centre at (0.5, 0.5, 0.5)
    // from a default orbit position outside the cube so the BackSide
    // material's first hit is the back face (the standard, robust setup).
    const camera = new THREE.PerspectiveCamera(45, props.width / props.height, 0.01, 100);
    camera.position.set(1.5, 1.2, 1.8);
    camera.lookAt(0.5, 0.5, 0.5);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0.5, 0.5, 0.5);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    // Clamp the camera outside the cube. Diagonal of the unit cube is sqrt(3) ~ 1.73;
    // setting min 0.9 keeps us strictly outside any face but not so far as to feel zoomed out.
    controls.minDistance = 0.9;
    controls.maxDistance = 6.0;
    controls.update();

    const material = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader,
      fragmentShader,
      side: THREE.BackSide,             // run the fragment shader once per pixel covered by the cube
      transparent: true,
      depthTest: false,                 // volume composites against the existing background
      uniforms: {
        uPressure:      { value: null },
        uObstacle:      { value: null },
        uCameraPos:     { value: new THREE.Vector3() },
        uSteps:         { value: props.raySteps },
        uGamma:         { value: props.gamma },
        uAlphaScale:    { value: props.alphaScale },
        uObstacleAlpha: { value: props.obstacleAlpha },
      },
    });

    // Unit cube centred at (0.5, 0.5, 0.5). Texture coords map directly
    // to box-local space.
    const geo = new THREE.BoxGeometry(1, 1, 1);
    geo.translate(0.5, 0.5, 0.5);
    const cube = new THREE.Mesh(geo, material);
    scene.add(cube);

    // Bounding-box wireframe so the user can see the simulation cube even
    // when the field is mostly silence.
    const wf = new THREE.LineSegments(
      new THREE.WireframeGeometry(geo),
      new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.18 }),
    );
    scene.add(wf);

    // Driver markers go in their own group so we can clear / re-add per frame
    // without stomping on the rest of the scene.
    const driverGroup = new THREE.Group();
    scene.add(driverGroup);

    sceneRef.current = {
      renderer, scene, camera, controls, material, cube, driverGroup,
      pressureTex: null, obstacleTex: null,
      rafHandle: 0,
      pressureShapeKey: '', obstacleShapeKey: '',
    };

    // Render loop. Three.js needs a continuous loop because OrbitControls
    // damping animates over time; the per-frame texture upload is gated
    // inside the loop by ref equality.
    const tick = () => {
      const s = sceneRef.current;
      if (!s) return;
      s.controls.update();
      s.material.uniforms.uCameraPos.value.copy(s.camera.position);
      s.renderer.render(s.scene, s.camera);
      s.rafHandle = requestAnimationFrame(tick);
    };
    sceneRef.current.rafHandle = requestAnimationFrame(tick);

    return () => {
      const s = sceneRef.current;
      if (!s) return;
      cancelAnimationFrame(s.rafHandle);
      s.controls.dispose();
      s.renderer.dispose();
      s.material.dispose();
      cube.geometry.dispose();
      wf.geometry.dispose();
      (wf.material as THREE.Material).dispose();
      s.pressureTex?.dispose();
      s.obstacleTex?.dispose();
      mount.removeChild(s.renderer.domElement);
      sceneRef.current = null;
    };
    // Mount is one-time; subsequent prop changes feed in via the effects below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // -- Resize the renderer when width/height props change --------------
  useEffect(() => {
    const s = sceneRef.current;
    if (!s) return;
    s.renderer.setSize(props.width, props.height);
    s.camera.aspect = props.width / props.height;
    s.camera.updateProjectionMatrix();
  }, [props.width, props.height]);

  // -- Push uniforms when the transfer-function knobs change -----------
  useEffect(() => {
    const s = sceneRef.current;
    if (!s) return;
    s.material.uniforms.uGamma.value = props.gamma;
    s.material.uniforms.uAlphaScale.value = props.alphaScale;
    s.material.uniforms.uObstacleAlpha.value = props.obstacleAlpha;
    s.material.uniforms.uSteps.value = props.raySteps;
  }, [props.gamma, props.alphaScale, props.obstacleAlpha, props.raySteps]);

  // -- Reset camera on the imperative nonce ----------------------------
  useEffect(() => {
    const s = sceneRef.current;
    if (!s) return;
    s.camera.position.set(1.5, 1.2, 1.8);
    s.controls.target.set(0.5, 0.5, 0.5);
    s.controls.update();
  }, [props.resetCameraNonce]);

  // -- Pressure texture upload (gated by the frame counter) ------------
  useEffect(() => {
    const s = sceneRef.current;
    if (!s) return;
    const buf = props.pressureRef.current;
    if (!buf || !buf.bytes || !buf.shape) return;
    const [nx, ny, nz] = buf.shape;
    const expected = nx * ny * nz;
    if (buf.bytes.length !== expected) {
      console.warn(`pressure bytes ${buf.bytes.length} != shape product ${expected}`);
      return;
    }
    const shapeKey = `${nx}x${ny}x${nz}`;
    if (s.pressureTex === null || s.pressureShapeKey !== shapeKey) {
      s.pressureTex?.dispose();
      const tex = new THREE.Data3DTexture(buf.bytes, nx, ny, nz);
      tex.format = THREE.RedFormat;
      tex.type = THREE.UnsignedByteType;
      tex.minFilter = THREE.LinearFilter;
      tex.magFilter = THREE.LinearFilter;
      tex.wrapR = tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
      tex.unpackAlignment = 1;
      tex.needsUpdate = true;
      s.pressureTex = tex;
      s.pressureShapeKey = shapeKey;
      s.material.uniforms.uPressure.value = tex;
    } else {
      // Same shape — swap the data array and mark dirty.
      (s.pressureTex.image as { data: Uint8Array }).data = buf.bytes;
      s.pressureTex.needsUpdate = true;
    }
    // We deliberately depend on the frame counter, not on buf.bytes,
    // so the typed array never enters React's prop-tracking path.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.pressureFrame]);

  // -- Obstacle texture upload (gated by the obstacle frame counter) ---
  useEffect(() => {
    const s = sceneRef.current;
    if (!s) return;
    const buf = props.obstacleRef.current;
    if (!buf || !buf.bytes || !buf.shape) return;
    const [nx, ny, nz] = buf.shape;
    const expected = nx * ny * nz;
    if (buf.bytes.length !== expected) {
      console.warn(`obstacle bytes ${buf.bytes.length} != shape product ${expected}`);
      return;
    }
    const shapeKey = `${nx}x${ny}x${nz}`;
    if (s.obstacleTex === null || s.obstacleShapeKey !== shapeKey) {
      s.obstacleTex?.dispose();
      const tex = new THREE.Data3DTexture(buf.bytes, nx, ny, nz);
      tex.format = THREE.RedFormat;
      tex.type = THREE.UnsignedByteType;
      tex.minFilter = THREE.NearestFilter;
      tex.magFilter = THREE.NearestFilter;
      tex.wrapR = tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
      tex.unpackAlignment = 1;
      tex.needsUpdate = true;
      s.obstacleTex = tex;
      s.obstacleShapeKey = shapeKey;
      s.material.uniforms.uObstacle.value = tex;
    } else {
      (s.obstacleTex.image as { data: Uint8Array }).data = buf.bytes;
      s.obstacleTex.needsUpdate = true;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.obstacleFrame]);

  // -- Driver markers (refresh whenever the driver list changes) -------
  useEffect(() => {
    const s = sceneRef.current;
    if (!s) return;
    // Wipe the previous markers — the driver list is small (~handful), so
    // recreating the meshes is cheaper than diff-ing.
    while (s.driverGroup.children.length) {
      const m = s.driverGroup.children.pop()! as THREE.Mesh;
      m.geometry.dispose();
      (m.material as THREE.Material).dispose();
    }
    if (!props.fullGridShape) return;
    const [Nx, Ny, Nz] = props.fullGridShape;
    const geo = new THREE.SphereGeometry(0.012, 16, 16);
    const mat = new THREE.MeshBasicMaterial({ color: 0xffc800 });
    for (const d of props.drivers) {
      const [i, j, k] = d.position;
      const m = new THREE.Mesh(geo.clone(), mat.clone());
      // Map full-grid (i, j, k) -> box-local (x, y, z) in [0, 1]^3.
      m.position.set((i + 0.5) / Nx, (j + 0.5) / Ny, (k + 0.5) / Nz);
      s.driverGroup.add(m);
    }
    return () => {
      geo.dispose();
      mat.dispose();
    };
  }, [props.drivers, props.fullGridShape]);

  return (
    <div
      ref={mountRef}
      style={{ width: props.width, height: props.height, background: '#0a0d12' }}
    />
  );
};

export default Volume;
