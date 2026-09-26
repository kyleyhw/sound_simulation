import{_ as e}from"./index-DHVGpwdT.js";var t=256,n=`
struct Params {
  nx: u32, ny: u32, nz: u32, dims: u32,
  n: u32, cpmlAxes: u32, murMask: u32, np: u32,
  nd: u32, off0: u32, off1: u32, off2: u32,
  murK: f32, pad0: f32, pad1: f32, pad2: f32,
};
@group(0) @binding(0) var<uniform> P: Params;

fn coords(idx: u32) -> vec3<u32> {
  let z = idx % P.nz;
  let j = (idx / P.nz) % P.ny;
  let i = idx / (P.nz * P.ny);
  return vec3<u32>(i, j, z);
}
fn stride(a: u32) -> u32 {
  if (a == 0u) { return P.ny * P.nz; }
  if (a == 1u) { return P.nz; }
  return 1u;
}
fn extent(a: u32) -> u32 {
  if (a == 0u) { return P.nx; }
  if (a == 1u) { return P.ny; }
  return P.nz;
}
fn profOff(a: u32) -> u32 {
  if (a == 0u) { return P.off0; }
  if (a == 1u) { return P.off1; }
  return P.off2;
}
fn flatId(g: vec3<u32>, n: vec3<u32>) -> u32 { return g.x + g.y * n.x * ${t}u; }
`,r=`${n}
@group(0) @binding(1) var<storage, read> p: array<f32>;
@group(0) @binding(2) var<storage, read> pp: array<f32>;
@group(0) @binding(3) var<storage, read_write> pn: array<f32>;
@group(0) @binding(4) var<storage, read> coefA: array<vec4<f32>>;   // C, S, Q, Qa
@group(0) @binding(5) var<storage, read> coefB: array<vec4<f32>>;   // InvA, Ks, flags, dt
@group(0) @binding(6) var<storage, read_write> st: array<vec2<f32>>; // V, X
@group(0) @binding(7) var<storage, read> ext: array<f32>;

@compute @workgroup_size(${t})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let idx = flatId(gid, nw);
  if (idx >= P.n) { return; }
  let b = coefB[idx];
  let flags = u32(b.z);
  if ((flags & 1u) == 0u) { pn[idx] = 0.0; return; }
  let K = f32((flags >> 1u) & 15u);
  let c = coords(idx);
  let pc = p[idx];
  var acc = 0.0;
  if (c.x > 0u) { acc += p[idx - stride(0u)]; }
  if (c.x < P.nx - 1u) { acc += p[idx + stride(0u)]; }
  if (c.y > 0u) { acc += p[idx - stride(1u)]; }
  if (c.y < P.ny - 1u) { acc += p[idx + stride(1u)]; }
  if (P.dims == 3u) {
    if (c.z > 0u) { acc += p[idx - 1u]; }
    if (c.z < P.nz - 1u) { acc += p[idx + 1u]; }
  }
  var lap = acc - K * pc;
  if (P.cpmlAxes != 0u) { lap += ext[idx]; }
  let a = coefA[idx];
  let sd = a.y;
  var rhs = 2.0 * pc - pp[idx] + a.x * lap + sd * pp[idx];
  let q = a.w;
  if (q != 0.0) {
    let s = st[idx];
    rhs += -q * (0.5 * pc - b.y * s.y) + a.z * s.x;
    let nxt = rhs / (1.0 + 0.5 * q + sd);
    let vn = (0.5 * (nxt + pc) - b.y * s.y) * b.x;
    st[idx] = vec2<f32>(vn, s.y + b.w * vn);
    pn[idx] = nxt;
  } else {
    pn[idx] = rhs / (1.0 + sd);
  }
}
`,i=`${n}
@group(0) @binding(1) var<storage, read> p: array<f32>;
@group(0) @binding(2) var<storage, read_write> psi: array<f32>;
@group(0) @binding(3) var<storage, read> coefB: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> prof: array<f32>;

fn wallAt(q: u32) -> bool { return ((u32(coefB[q].z) >> 5u) & 1u) == 1u; }

@compute @workgroup_size(${t})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let idx = flatId(gid, nw);
  if (idx >= P.n) { return; }
  let c = coords(idx);
  for (var a = 0u; a < P.dims; a++) {
    if (((P.cpmlAxes >> a) & 1u) == 0u) { continue; }
    let h = select(select(c.z, c.y, a == 1u), c.x, a == 0u);
    let na = extent(a);
    if (h + 1u >= na) { continue; }
    let s = stride(a);
    var d = p[idx + s] - p[idx];
    if (wallAt(idx) || wallAt(idx + s)) { d = 0.0; }
    let o = profOff(a);
    let ah = prof[o + 2u * na + h];
    let bh = prof[o + 3u * na + h];
    let k = a * P.n + idx;
    psi[k] = bh * psi[k] + ah * d;
  }
}
`,a=`${n}
@group(0) @binding(1) var<storage, read> p: array<f32>;
@group(0) @binding(2) var<storage, read> psi: array<f32>;
@group(0) @binding(3) var<storage, read_write> zeta: array<f32>;
@group(0) @binding(4) var<storage, read_write> ext: array<f32>;
@group(0) @binding(5) var<storage, read> coefB: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> prof: array<f32>;

fn wallAt(q: u32) -> bool { return ((u32(coefB[q].z) >> 5u) & 1u) == 1u; }

@compute @workgroup_size(${t})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let idx = flatId(gid, nw);
  if (idx >= P.n) { return; }
  let c = coords(idx);
  var e = 0.0;
  for (var a = 0u; a < P.dims; a++) {
    if (((P.cpmlAxes >> a) & 1u) == 0u) { continue; }
    let i = select(select(c.z, c.y, a == 1u), c.x, a == 0u);
    let na = extent(a);
    if (i == 0u || i + 1u >= na) { continue; }
    let s = stride(a);
    var up = p[idx + s] - p[idx];
    if (wallAt(idx) || wallAt(idx + s)) { up = 0.0; }
    var dn = p[idx] - p[idx - s];
    if (wallAt(idx - s) || wallAt(idx)) { dn = 0.0; }
    let k = a * P.n + idx;
    let dpsi = psi[k] - psi[k - s];
    let o = profOff(a);
    let an = prof[o + i];
    let bn = prof[o + na + i];
    let z = bn * zeta[k] + an * (up - dn + dpsi);
    zeta[k] = z;
    e += dpsi + z;
  }
  ext[idx] = e;
}
`,o=`${n}
override AXIS: u32 = 0u;
@group(0) @binding(1) var<storage, read> p: array<f32>;
@group(0) @binding(2) var<storage, read_write> pn: array<f32>;

@compute @workgroup_size(${t})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let idx = flatId(gid, nw);
  if (idx >= P.n) { return; }
  let c = coords(idx);
  let i = select(select(c.z, c.y, AXIS == 1u), c.x, AXIS == 0u);
  let na = extent(AXIS);
  let s = stride(AXIS);
  // High face after low face, as in the CPU loop.
  if (((P.murMask >> (2u * AXIS + 1u)) & 1u) == 1u && i == na - 1u) {
    pn[idx] = p[idx - s] + P.murK * (pn[idx - s] - p[idx]);
  } else if (((P.murMask >> (2u * AXIS)) & 1u) == 1u && i == 0u) {
    pn[idx] = p[idx + s] + P.murK * (pn[idx + s] - p[idx]);
  }
}
`,s=`${n}
@group(0) @binding(1) var<storage, read_write> pn: array<f32>;
@group(0) @binding(2) var<storage, read> drvIdx: array<u32>;
@group(0) @binding(3) var<storage, read> drvVal: array<f32>;
@group(0) @binding(4) var<storage, read> counter: array<u32>;

@compute @workgroup_size(1)
fn main() {
  let k = counter[0];
  for (var j = 0u; j < P.nd; j++) { pn[drvIdx[j]] += drvVal[k * P.nd + j]; }
}
`,c=`${n}
@group(0) @binding(1) var<storage, read> pn: array<f32>;
@group(0) @binding(2) var<storage, read> probeIdx: array<u32>;
@group(0) @binding(3) var<storage, read_write> probeOut: array<f32>;
@group(0) @binding(4) var<storage, read_write> counter: array<u32>;

@compute @workgroup_size(1)
fn main() {
  let k = counter[0];
  for (var j = 0u; j < P.np; j++) { probeOut[k * P.np + j] = pn[probeIdx[j]]; }
  counter[0] = k + 1u;
}
`,l=`${n}
@group(0) @binding(1) var<storage, read> pn: array<f32>;
@group(0) @binding(2) var<storage, read_write> acc: array<f32>;

@compute @workgroup_size(${t})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let idx = flatId(gid, nw);
  if (idx >= P.n) { return; }
  acc[idx] += pn[idx] * pn[idx];
}
`;async function u(){try{return!!(typeof navigator<`u`&&navigator.gpu&&await navigator.gpu.requestAdapter())}catch{return!1}}var d=class n{device;sim;bufs={};p3=[];rot=0;pipes={};batchCap=0;geometryKey=null;cpmlAxes=0;murMask=0;dispatch=[1,1];busy=!1;needUpload=!1;needRmsZero=!1;constructor(e,t){this.device=e,this.sim=t}static async create(e){let t=await navigator.gpu?.requestAdapter();if(!t)throw Error(`WebGPU is not available in this browser`);let r=e.n*4*3,i=await t.requestDevice({requiredLimits:{maxStorageBufferBindingSize:Math.min(t.limits.maxStorageBufferBindingSize,Math.max(r,128<<20)),maxBufferSize:Math.min(t.limits.maxBufferSize,Math.max(r,256<<20))}}),a=new n(i,e);return a.build(),a.upload(),e.onReset=()=>a.needUpload=!0,e.onResetAccumulators=()=>a.needRmsZero=!0,a}buffer(e,t,n){this.bufs[e]?.destroy();let r=this.device.createBuffer({size:Math.max(16,Math.ceil(t/4)*4),usage:n});return this.bufs[e]=r,r}pipeline(e,t,n){let r=n?`${e}:${JSON.stringify(n)}`:e;return this.pipes[r]||(this.pipes[r]=this.device.createComputePipeline({layout:`auto`,compute:{module:this.device.createShaderModule({code:t}),entryPoint:`main`,constants:n}})),this.pipes[r]}build(){let{n:e}=this.sim,n=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;this.p3=[0,1,2].map(t=>this.buffer(`p${t}`,e*4,n)),this.buffer(`coefA`,e*16,n),this.buffer(`coefB`,e*16,n),this.buffer(`state`,e*8,n),this.buffer(`ext`,e*4,n),this.buffer(`psi`,3*e*4,n),this.buffer(`zeta`,3*e*4,n),this.buffer(`counter`,16,n),this.buffer(`params`,64,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);let r=Math.ceil(e/t),i=Math.min(r,65535);this.dispatch=[i,Math.ceil(r/i)]}upload(){let e=this.sim,t=e.deviceState(),n=e.n,r=this.device.queue,i=new Float32Array(4*n),a=new Float32Array(4*n),o=t.cpml?.wall??null;for(let r=0;r<n;r++)i[4*r]=t.C[r],i[4*r+1]=t.S[r],i[4*r+2]=t.Q[r],i[4*r+3]=t.Qa[r],a[4*r]=t.InvA[r],a[4*r+1]=t.Ks[r],a[4*r+2]=t.active[r]+2*t.K[r]+32*(o&&o[r]?1:0),a[4*r+3]=e.dt;r.writeBuffer(this.bufs.coefA,0,i),r.writeBuffer(this.bufs.coefB,0,a);let s=new Float32Array(2*n);for(let e=0;e<n;e++)s[2*e]=t.V[e],s[2*e+1]=t.X[e];r.writeBuffer(this.bufs.state,0,s),this.rot=0,r.writeBuffer(this.p3[0],0,e.p),r.writeBuffer(this.p3[2],0,e.pPrev);let c=[e.nx,e.ny,e.nz],l=[0,0,0];this.cpmlAxes=0;let u=new Float32Array(3*n),d=new Float32Array(3*n),f=new Float32Array(4);if(t.cpml){let i=0;for(let n=0;n<e.dims;n++)l[n]=i,t.cpml.axes[n]&&(i+=4*c[n]);f=new Float32Array(Math.max(4,i)),t.cpml.axes.forEach((e,t)=>{if(!e)return;this.cpmlAxes|=1<<t;let r=l[t];f.set(e.an,r),f.set(e.bn,r+c[t]),f.set(e.ah,r+2*c[t]),f.set(e.bh,r+3*c[t]),u.set(e.psi,t*n),d.set(e.zeta,t*n)}),r.writeBuffer(this.bufs.ext,0,new Float32Array(n))}this.buffer(`prof`,f.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),r.writeBuffer(this.bufs.prof,0,f),r.writeBuffer(this.bufs.psi,0,u),r.writeBuffer(this.bufs.zeta,0,d),this.murMask=t.murFaces.reduce((e,t,n)=>t?e|1<<n:e,0);let p=new ArrayBuffer(64),m=new Uint32Array(p),h=new Float32Array(p);m.set([e.nx,e.ny,e.nz,e.dims,n,this.cpmlAxes,this.murMask,0,0,l[0],l[1],l[2]]),h[12]=(t.lam-1)/(t.lam+1),this.paramsHost=p,r.writeBuffer(this.bufs.params,0,p),this.geometryKey=t.geometryKey,e.rmsAccum?(this.buffer(`rms`,n*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC),r.writeBuffer(this.bufs.rms,0,e.rmsAccum)):(this.bufs.rms?.destroy(),delete this.bufs.rms)}paramsHost=new ArrayBuffer(64);group(e,t){return this.device.createBindGroup({layout:e.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.bufs.params}},...t.map((e,t)=>({binding:t+1,resource:{buffer:e}}))]})}get stale(){return this.sim.deviceState().geometryKey!==this.geometryKey||!!this.sim.rmsAccum!=!!this.bufs.rms}async run(e,t=!1){this.needUpload&&(this.needUpload=!1,this.needRmsZero=!1,this.upload()),this.needRmsZero&&this.bufs.rms&&(this.needRmsZero=!1,this.device.queue.writeBuffer(this.bufs.rms,0,new Float32Array(this.sim.n))),this.stale&&(await this.advance(0,!0),this.upload()),await this.advance(e,t)}async advance(t,n){let u=this.sim,d=u.n,f=u.drivers.filter(e=>e.enabled),p=f.length,m=u.probes.length,h=new Float32Array(Math.max(1,t*p)),g=u.time;for(let n=0;n<t;n++){for(let t=0;t<p;t++){let r=f[t];h[n*p+t]=e(r.waveform,g-(r.delay??0))*(r.gain??1)}g+=u.dt}let _=GPUBufferUsage;(t>this.batchCap||!this.bufs.drvVal)&&(this.batchCap=Math.max(t,64),this.buffer(`drvVal`,this.batchCap*Math.max(1,p)*4*4,_.STORAGE|_.COPY_DST),this.buffer(`probeOut`,this.batchCap*Math.max(1,m)*4*4,_.STORAGE|_.COPY_SRC|_.COPY_DST)),this.bufs.drvVal.size<h.byteLength&&this.buffer(`drvVal`,h.byteLength,_.STORAGE|_.COPY_DST),this.bufs.probeOut.size<t*Math.max(1,m)*4&&this.buffer(`probeOut`,t*Math.max(1,m)*4,_.STORAGE|_.COPY_SRC|_.COPY_DST);let v=this.device.queue;v.writeBuffer(this.bufs.drvVal,0,h),this.buffer(`drvIdx`,Math.max(1,p)*4,_.STORAGE|_.COPY_DST),v.writeBuffer(this.bufs.drvIdx,0,new Uint32Array(p?f.map(e=>u.index(e.pos)):[0])),this.buffer(`probeIdx`,Math.max(1,m)*4,_.STORAGE|_.COPY_DST),v.writeBuffer(this.bufs.probeIdx,0,new Uint32Array(m?u.probes.map(e=>u.index(e.pos)):[0])),v.writeBuffer(this.bufs.counter,0,new Uint32Array([0,0,0,0]));let y=new Uint32Array(this.paramsHost);y[7]=m,y[8]=p,v.writeBuffer(this.bufs.params,0,this.paramsHost);let b={general:this.pipeline(`general`,r),psi:this.pipeline(`psi`,i),zeta:this.pipeline(`zeta`,a),inject:this.pipeline(`inject`,s),probe:this.pipeline(`probe`,c),rms:this.pipeline(`rms`,l),mur:[0,1,2].map(e=>this.pipeline(`mur`,o,{AXIS:e}))},x=[0,1,2].map(e=>{let t=this.p3[e],n=this.p3[(e+2)%3],r=this.p3[(e+1)%3],i=this.bufs;return{general:this.group(b.general,[t,n,r,i.coefA,i.coefB,i.state,i.ext]),psi:this.cpmlAxes?this.group(b.psi,[t,i.psi,i.coefB,i.prof]):null,zeta:this.cpmlAxes?this.group(b.zeta,[t,i.psi,i.zeta,i.ext,i.coefB,i.prof]):null,mur:b.mur.map((e,n)=>this.murMask>>2*n&3?this.group(e,[t,r]):null),inject:p?this.group(b.inject,[r,i.drvIdx,i.drvVal,i.counter]):null,probe:this.group(b.probe,[r,i.probeIdx,i.probeOut,i.counter]),rms:i.rms?this.group(b.rms,[r,i.rms]):null}}),S=this.device.createCommandEncoder(),C=S.beginComputePass(),[w,T]=this.dispatch;for(let e=0;e<t;e++){let e=x[this.rot];e.psi&&e.zeta&&(C.setPipeline(b.psi),C.setBindGroup(0,e.psi),C.dispatchWorkgroups(w,T),C.setPipeline(b.zeta),C.setBindGroup(0,e.zeta),C.dispatchWorkgroups(w,T)),C.setPipeline(b.general),C.setBindGroup(0,e.general),C.dispatchWorkgroups(w,T),e.mur.forEach((e,t)=>{e&&(C.setPipeline(b.mur[t]),C.setBindGroup(0,e),C.dispatchWorkgroups(w,T))}),e.inject&&(C.setPipeline(b.inject),C.setBindGroup(0,e.inject),C.dispatchWorkgroups(1)),C.setPipeline(b.probe),C.setBindGroup(0,e.probe),C.dispatchWorkgroups(1),e.rms&&(C.setPipeline(b.rms),C.setBindGroup(0,e.rms),C.dispatchWorkgroups(w,T)),this.rot=(this.rot+1)%3}C.end();let E=[[this.p3[this.rot],d*4],[this.p3[(this.rot+2)%3],d*4],[this.bufs.probeOut,Math.max(1,t*m)*4]];this.bufs.rms&&E.push([this.bufs.rms,d*4]),n&&E.push([this.bufs.state,d*8],[this.bufs.psi,3*d*4],[this.bufs.zeta,3*d*4]);let D=E.map(([e,t])=>{let n=this.device.createBuffer({size:t,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});return S.copyBufferToBuffer(e,0,n,0,t),n});v.submit([S.finish()]),await Promise.all(D.map(e=>e.mapAsync(GPUMapMode.READ)));let O=D.map(e=>new Float32Array(e.getMappedRange().slice(0)));D.forEach(e=>e.destroy());let k=u.time;for(let e=0;e<t;e++)k+=u.dt;let A={p:O[0],pPrev:O[1],time:k,step_count:u.step_count+t},j=3;if(this.bufs.rms&&u.rmsAccum&&(u.rmsAccum.set(O[j++]),u.rmsCount+=t),n){let e=O[j++],t=new Float32Array(d),n=new Float32Array(d);for(let r=0;r<d;r++)t[r]=e[2*r],n[r]=e[2*r+1];A.V=t,A.X=n;let r=O[j++],i=O[j++],a=u.deviceState();(a.cpml?a.cpml.axes.reduce((e,t,n)=>t?e|1<<n:e,0):0)===this.cpmlAxes&&a.cpml?.axes.forEach((e,t)=>{e&&(e.psi.set(r.subarray(t*d,(t+1)*d)),e.zeta.set(i.subarray(t*d,(t+1)*d)))})}u.importState(A),m&&u.pushProbeSamples(O[2],t)}async syncState(){await this.advance(0,!0)}destroy(){this.sim.onReset&&(this.sim.onReset=void 0),this.sim.onResetAccumulators&&(this.sim.onResetAccumulators=void 0),Object.values(this.bufs).forEach(e=>e.destroy()),this.p3.forEach(e=>e.destroy()),this.device.destroy()}};export{d as GpuStepper,u as gpuAvailable};
//# sourceMappingURL=gpu-DVm_kMjw.js.map