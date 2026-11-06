import * as THREE from 'three';
import { triTable } from 'three/addons/objects/MarchingCubes.js';

// ---------------------------------------------------------------------------
// Surface skinning for the particle sim. Three interchangeable techniques, all
// driven from the sim's pos/flags buffers, plus the existing sphere impostors:
//
//   'spheres'  — handled by Renderer (the raw impostor path); not here.
//   'mc'       — marching cubes: splat particles into a per-group density grid,
//                extract an isosurface mesh on the GPU (three's triTable), draw
//                it as real lit geometry.
//   'raymarch' — sphere-trace the same density grid in a fullscreen pass.
//   'ssfr'     — screen-space fluids: particle depth → bilateral blur → normal
//                reconstruction → refractive/opaque composite.
//
// The density grid (MC + raymarch) stores three channels per voxel — red body,
// green body, water — as fixed-point atomic counts, so the surface knows which
// material it belongs to and water can be shaded as a refractive liquid while
// the bodies stay opaque.
// ---------------------------------------------------------------------------

const WSCALE = 4096;            // fixed-point scale for atomic density accum
const MC_MAX_VERTS = 1_500_000; // marching-cubes vertex buffer capacity

// group colours (must match the impostor materials in render.js)
const COL_RED = [0.88, 0.12, 0.2];
const COL_GREEN = [0.15, 0.78, 0.45];
const COL_WATER = [0.15, 0.5, 0.92];

// WGSL snippet shared by every skin shader: the uniform block + the density
// grid sampling helpers. `dens` is declared per-shader (atomic vs plain).
const COMMON = /* wgsl */ `
struct Uniforms {
  view: mat4x4f,
  proj: mat4x4f,
  invProj: mat4x4f,
  invView: mat4x4f,
  camPos: vec4f,
  lightView: vec4f,
  screen: vec4f,      // w, h, 1/w, 1/h
  gridMin: vec4f,     // xyz origin, w = voxel size
  gridDim: vec4u,     // nx, ny, nz
  counts: vec4u,      // end of body-0 balls, end of balls, TOTAL, first ball
  p0: vec4f,          // iso, waterR, bodyR, _
  p1: vec4f,          // absorb, refractScale, fresnelPow, hqFlag
  p2: vec4f,          // texMode, spacing, tearFactor, waterOnly
};
const COL = array<vec3f,3>(
  vec3f(${COL_RED.join(',')}),
  vec3f(${COL_GREEN.join(',')}),
  vec3f(${COL_WATER.join(',')}));
`;

// trilinear density read (plain read-only grid) — returns the three channel
// densities at a world point; total = .x+.y+.z is the scalar field.
const SAMPLE = /* wgsl */ `
fn cellCh(ix: i32, iy: i32, iz: i32) -> vec3f {
  let nx = i32(U.gridDim.x); let ny = i32(U.gridDim.y); let nz = i32(U.gridDim.z);
  if (ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz) { return vec3f(0.0); }
  let idx = (((iz * ny) + iy) * nx + ix) * 4;
  return vec3f(f32(dens[idx]), f32(dens[idx + 1]), f32(dens[idx + 2])) / ${WSCALE}.0;
}
fn sampleCh(wp: vec3f) -> vec3f {
  let gi = (wp - U.gridMin.xyz) / U.gridMin.w;
  let b = floor(gi);
  let f = gi - b;
  let ix = i32(b.x); let iy = i32(b.y); let iz = i32(b.z);
  let c000 = cellCh(ix, iy, iz);     let c100 = cellCh(ix + 1, iy, iz);
  let c010 = cellCh(ix, iy + 1, iz); let c110 = cellCh(ix + 1, iy + 1, iz);
  let c001 = cellCh(ix, iy, iz + 1);     let c101 = cellCh(ix + 1, iy, iz + 1);
  let c011 = cellCh(ix, iy + 1, iz + 1); let c111 = cellCh(ix + 1, iy + 1, iz + 1);
  let x00 = mix(c000, c100, f.x); let x10 = mix(c010, c110, f.x);
  let x01 = mix(c001, c101, f.x); let x11 = mix(c011, c111, f.x);
  let y0 = mix(x00, x10, f.y); let y1 = mix(x01, x11, f.y);
  return mix(y0, y1, f.z);
}
fn fieldAt(wp: vec3f) -> f32 { let c = sampleCh(wp); return c.x + c.y + c.z; }
fn gradAtH(wp: vec3f, h: f32) -> vec3f {
  return vec3f(
    fieldAt(wp + vec3f(h,0,0)) - fieldAt(wp - vec3f(h,0,0)),
    fieldAt(wp + vec3f(0,h,0)) - fieldAt(wp - vec3f(0,h,0)),
    fieldAt(wp + vec3f(0,0,h)) - fieldAt(wp - vec3f(0,0,h)));
}
fn gradAt(wp: vec3f) -> vec3f { return gradAtH(wp, U.gridMin.w); }
fn colorAt(wp: vec3f) -> vec4f {
  let c = sampleCh(wp);
  let t = max(c.x + c.y + c.z, 1e-4);
  let rgb = (c.x * COL[0] + c.y * COL[1] + c.z * COL[2]) / t;
  return vec4f(rgb, c.z / t); // a = water fraction
}
`;

// procedural texture (no external assets): checker + grooves + value noise,
// plus a screen-space-derivative bump that turns the height into a normal
// perturbation — works for UV meshes and for triplanar/world-space sampling.
const PROC = /* wgsl */ `
fn hash21(p: vec2f) -> f32 {
  return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}
fn vnoise(p: vec2f) -> f32 {
  let i = floor(p); let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = hash21(i); let b = hash21(i + vec2f(1.0, 0.0));
  let c = hash21(i + vec2f(0.0, 1.0)); let d = hash21(i + vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
// returns albedo.rgb (tinted by base) and height in .a
fn procSurf(uv: vec2f, base: vec3f) -> vec4f {
  let scale = 5.0;
  let g = uv * scale;
  let cell = floor(g);
  let s = cell.x + cell.y;
  let parity = s - 2.0 * floor(s * 0.5);            // 0 / 1 checker
  let checker = mix(0.82, 1.18, parity);
  let fp = abs(fract(g) - 0.5);
  let line = smoothstep(0.42, 0.5, max(fp.x, fp.y)); // grooves between tiles
  let n = vnoise(uv * scale * 2.0) * 0.3 + 0.8;
  var tint = base * checker * n;
  tint = mix(tint, base * 0.35, line);
  let height = (1.0 - line) * (0.55 + 0.45 * n);
  return vec4f(tint, height);
}
// Mikkelsen screen-space surface-gradient bump (no precomputed tangents)
fn bumpNormal(N: vec3f, P: vec3f, h: f32, strength: f32) -> vec3f {
  let dpx = dpdx(P); let dpy = dpdy(P);
  let dhx = dpdx(h); let dhy = dpdy(h);
  let r1 = cross(dpy, N); let r2 = cross(N, dpx);
  let det = dot(dpx, r1);
  if (abs(det) < 1e-12) { return N; }
  let grad = (dhx * r1 + dhy * r2) / det;
  return normalize(N - strength * grad);
}
// triplanar procedural sample: blend three planar projections by world normal
fn triplanar(P: vec3f, N: vec3f, base: vec3f) -> vec4f {
  var w = abs(N);
  w = w / (w.x + w.y + w.z);
  let xz = procSurf(P.xz, base);
  let xy = procSurf(P.xy, base);
  let zy = procSurf(P.zy, base);
  return zy * w.x + xz * w.y + xy * w.z;
}
`;

// fullscreen-triangle vertex stage (shared by raymarch / blit / ssfr composite)
const FULLSCREEN_VS = /* wgsl */ `
struct VOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn fsv(@builtin(vertex_index) vi: u32) -> VOut {
  var p = array<vec2f,3>(vec2f(-1.0,-1.0), vec2f(3.0,-1.0), vec2f(-1.0,3.0));
  var o: VOut;
  o.pos = vec4f(p[vi], 0.0, 1.0);
  o.uv = vec2f((p[vi].x + 1.0) * 0.5, (1.0 - p[vi].y) * 0.5);
  return o;
}
`;

export class Skin {
  constructor(device, sim, format) {
    this.device = device;
    this.format = format;
    this.depthFmt = 'depth24plus';
    this.w = 0; this.h = 0;

    // uniform mirror: 100 f32 = 400 bytes (see struct Uniforms)
    this.ub = device.createBuffer({ size: 400, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.uf = new Float32Array(100);
    this.uu = new Uint32Array(this.uf.buffer);

    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

    // marching-cubes triangle table (edge ids, -1 terminated), 256*16 entries
    this.triBuf = device.createBuffer({
      size: triTable.length * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.triBuf, 0, new Int32Array(triTable));

    this._buildPipelines();
    this.attachSim(sim);
  }

  // (re)bind to a sim: grid sizing + per-group descriptors + density buffers
  attachSim(sim) {
    this.sim = sim;
    const p = sim.o.params;
    // a fixed world box around the play area, padded above for grabbed bodies
    const minB = [-p.wallX - 0.4, -0.15, -p.wallZ - 0.4];
    const maxB = [p.wallX + 0.4, 5.5, p.wallZ + 0.4];
    const voxel = Math.min(0.2, Math.max(0.045, p.fluidH * 0.5));
    this.voxel = voxel;
    this.gridMin = minB;
    this.dim = maxB.map((m, i) => Math.min(224, Math.ceil((m - minB[i]) / voxel) + 1));
    const nVox = this.dim[0] * this.dim[1] * this.dim[2];

    if (this.densBuf) this.densBuf.destroy();
    this.densBuf = this.device.createBuffer({
      size: nVox * 4 * 4, // 4 channels (r,g,water,pad) * u32
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.nVox = nVox;

    // material-space (#3): per-voxel weighted body rest-position accumulation
    // (i32 fixed-point), divided by body density at sample time → a material UVW
    // that deforms with the body. Plus a buffer of each body particle's rest pos.
    if (this.restBuf) this.restBuf.destroy();
    this.restBuf = this.device.createBuffer({
      size: nVox * 4 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (this.restPosBuf) this.restPosBuf.destroy();
    const rp = new Float32Array(sim.SOLID_N * 4);
    rp.set(sim.init.pos.subarray(0, sim.SOLID_N * 4));
    this.restPosBuf = this.device.createBuffer({
      size: rp.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.restPosBuf, 0, rp);

    // group descriptors: base index, count, channel, colour, impostor radius, water flag
    const P = sim.o.params;
    this.groups = [
      { base: sim.BALL0, count: sim.N1 - sim.BALL0, g: 0, color: COL_RED, radius: P.solidRadius * 1.3, water: 0 },
      { base: sim.N1, count: sim.SOLID_N - sim.N1, g: 1, color: COL_GREEN, radius: P.solidRadius * 1.3, water: 0 },
      { base: sim.SOLID_N, count: sim.o.MAX_FLUID, g: 2, color: COL_WATER, radius: P.fluidRadius * 1.8, water: 1 },
    ];

    // splat smoothing radii (per group). water settles densely so a tight
    // kernel keeps it crisp; the body lattice is sparser, so it needs a wider
    // kernel (relative to its spacing) to fuse neighbours into a solid skin.
    this.waterR = P.fluidH * 0.9;
    // metaballs at the voxel centres, radius 1.85 lattice units — the value
    // that looked best in docs/disphenoid-voxel-builder.html (threshold 1.5)
    this.bodyR = sim.o.spacing * 1.85;

    this._buildSimBindGroups();
    this._buildEmbedMeshes();
  }

  _buildPipelines() {
    const d = this.device;

    // ---- bind group layouts ----
    // uniform-only (group 0 for fullscreen passes that also take textures)
    this.uboBGL = d.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }],
    });

    // density as atomic (clear + splat)
    this.densAtomicBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },        // dens (atomic)
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // pos
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // flags
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },        // restGrid (atomic i32)
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // restPos
      ],
    });

    // density read-only (mc compute + raymarch fragment); binding 2 = restGrid
    this.densReadBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      ],
    });

    // ===================== density: clear + splat =====================
    const splatMod = d.createShaderModule({ code: COMMON + /* wgsl */ `
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read_write> dens: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> pos: array<vec4f>;
@group(0) @binding(3) var<storage, read> flags: array<u32>;
@group(0) @binding(4) var<storage, read_write> restGrid: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read> restPos: array<vec4f>;
const RSCALE = 2048.0;

@compute @workgroup_size(64)
fn clearDens(@builtin(global_invocation_id) gid: vec3u,
             @builtin(num_workgroups) nwg: vec3u) {
  let total = U.gridDim.x * U.gridDim.y * U.gridDim.z * 4u;
  let stride = nwg.x * 64u;
  var i = gid.x;
  loop {
    if (i >= total) { break; }
    atomicStore(&dens[i], 0u);
    atomicStore(&restGrid[i], 0);
    i += stride;
  }
}

@compute @workgroup_size(64)
fn splat(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.counts.z || i < U.counts.w) { return; } // lattice vertices don't splat
  if ((flags[i] & 1u) == 0u) { return; }
  var g = 2u;
  if (i < U.counts.y) { g = 1u; }
  if (i < U.counts.x) { g = 0u; }
  if (U.p2.w > 0.5 && g != 2u) { return; } // embed mode: water-only field
  let p = pos[i].xyz;
  // body lattice is sparser than settled water, so it needs a wider kernel to
  // merge neighbours into a solid surface instead of per-particle spikes
  var R = U.p0.z;            // body radius
  if (g == 2u) { R = U.p0.y; } // water radius
  let voxel = U.gridMin.w;
  let nx = i32(U.gridDim.x); let ny = i32(U.gridDim.y); let nz = i32(U.gridDim.z);
  let gi = (p - U.gridMin.xyz) / voxel;
  let r = R / voxel;
  let lo = vec3i(floor(gi - r));
  let hi = vec3i(ceil(gi + r));
  let R2 = R * R;
  for (var z = lo.z; z <= hi.z; z++) {
    if (z < 0 || z >= nz) { continue; }
    for (var y = lo.y; y <= hi.y; y++) {
      if (y < 0 || y >= ny) { continue; }
      for (var x = lo.x; x <= hi.x; x++) {
        if (x < 0 || x >= nx) { continue; }
        let cw = U.gridMin.xyz + vec3f(f32(x), f32(y), f32(z)) * voxel;
        let d2 = dot(cw - p, cw - p);
        if (d2 >= R2) { continue; }
        let q = 1.0 - d2 / R2;
        let w = q * q * q;                 // poly6-ish falloff
        let vbase = (((z * ny) + y) * nx + x) * 4;
        atomicAdd(&dens[vbase + i32(g)], u32(w * ${WSCALE}.0));
        // material-space: accumulate weighted body rest position
        if (g != 2u) {
          let rp = restPos[i].xyz * (w * RSCALE);
          atomicAdd(&restGrid[vbase + 0], i32(rp.x));
          atomicAdd(&restGrid[vbase + 1], i32(rp.y));
          atomicAdd(&restGrid[vbase + 2], i32(rp.z));
        }
      }
    }
  }
}
`});
    const splatLayout = d.createPipelineLayout({ bindGroupLayouts: [this.densAtomicBGL] });
    this.pipeClear = d.createComputePipeline({ layout: splatLayout, compute: { module: splatMod, entryPoint: 'clearDens' } });
    this.pipeSplat = d.createComputePipeline({ layout: splatLayout, compute: { module: splatMod, entryPoint: 'splat' } });

    // ===================== marching cubes =====================
    this.mcBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // dens
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // triTable
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // verts
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // counter (atomic)
      ],
    });
    const mcMod = d.createShaderModule({ code: COMMON + SAMPLE + /* wgsl */ `
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> dens: array<u32>;
@group(0) @binding(2) var<storage, read> tri: array<i32>;
@group(0) @binding(3) var<storage, read_write> verts: array<f32>;
@group(0) @binding(4) var<storage, read_write> counter: array<atomic<u32>>;

@compute @workgroup_size(64)
fn clearCounter(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x == 0u) {
    atomicStore(&counter[0], 0u);  // vertex count
    atomicStore(&counter[1], 1u);  // instanceCount
    atomicStore(&counter[2], 0u);
    atomicStore(&counter[3], 0u);
  }
}

fn fieldCorner(ix: i32, iy: i32, iz: i32) -> f32 {
  let c = cellCh(ix, iy, iz); return c.x + c.y + c.z;
}

@compute @workgroup_size(4, 4, 4)
fn march(@builtin(global_invocation_id) gid: vec3u) {
  let x = i32(gid.x); let y = i32(gid.y); let z = i32(gid.z);
  if (x >= i32(U.gridDim.x) - 1 || y >= i32(U.gridDim.y) - 1 || z >= i32(U.gridDim.z) - 1) { return; }
  let iso = U.p0.x;

  // corner field values, three.js numbering
  var fv = array<f32,8>(
    fieldCorner(x,   y,   z),   fieldCorner(x+1, y,   z),
    fieldCorner(x,   y+1, z),   fieldCorner(x+1, y+1, z),
    fieldCorner(x,   y,   z+1), fieldCorner(x+1, y,   z+1),
    fieldCorner(x,   y+1, z+1), fieldCorner(x+1, y+1, z+1));
  var ci = 0u;
  if (fv[0] < iso) { ci |= 1u; }   if (fv[1] < iso) { ci |= 2u; }
  if (fv[2] < iso) { ci |= 8u; }   if (fv[3] < iso) { ci |= 4u; }
  if (fv[4] < iso) { ci |= 16u; }  if (fv[5] < iso) { ci |= 32u; }
  if (fv[6] < iso) { ci |= 128u; } if (fv[7] < iso) { ci |= 64u; }
  if (ci == 0u || ci == 255u) { return; }

  // corner offsets (field numbering) and the 12 edge corner-pairs
  var cp = array<vec3f,8>(
    vec3f(0,0,0), vec3f(1,0,0), vec3f(0,1,0), vec3f(1,1,0),
    vec3f(0,0,1), vec3f(1,0,1), vec3f(0,1,1), vec3f(1,1,1));
  var ea = array<i32,12>(0,1,2,0,4,5,6,4,0,1,3,2);
  var eb = array<i32,12>(1,3,3,2,5,7,7,6,4,5,7,6);

  let voxel = U.gridMin.w;
  let base = U.gridMin.xyz + vec3f(f32(x), f32(y), f32(z)) * voxel;
  let row = i32(ci) * 16;

  var i = 0;
  loop {
    let e0 = tri[row + i];
    if (e0 < 0) { break; }
    let e1 = tri[row + i + 1];
    let e2 = tri[row + i + 2];
    var es = array<i32,3>(e0, e1, e2);

    // reserve 3 vertices
    let vbase = atomicAdd(&counter[0], 3u);
    if (vbase + 3u > ${MC_MAX_VERTS}u) { break; }

    for (var k = 0; k < 3; k++) {
      let e = es[k];
      let a = ea[e]; let b = eb[e];
      let fa = fv[a]; let fb = fv[b];
      var t = 0.5;
      if (abs(fb - fa) > 1e-6) { t = clamp((iso - fa) / (fb - fa), 0.0, 1.0); }
      let pa = base + cp[a] * voxel;
      let pb = base + cp[b] * voxel;
      let wp = mix(pa, pb, t);
      let n = -normalize(gradAt(wp) + vec3f(1e-6));
      let col = colorAt(wp);
      let o = (vbase + u32(k)) * 9u;
      verts[o + 0] = wp.x; verts[o + 1] = wp.y; verts[o + 2] = wp.z;
      verts[o + 3] = n.x;  verts[o + 4] = n.y;  verts[o + 5] = n.z;
      verts[o + 6] = col.x; verts[o + 7] = col.y; verts[o + 8] = col.z;
    }
    i += 3;
  }
}
`});
    const mcLayout = d.createPipelineLayout({ bindGroupLayouts: [this.mcBGL] });
    this.pipeMcClear = d.createComputePipeline({ layout: mcLayout, compute: { module: mcMod, entryPoint: 'clearCounter' } });
    this.pipeMcMarch = d.createComputePipeline({ layout: mcLayout, compute: { module: mcMod, entryPoint: 'march' } });

    // mc draw pipeline (real geometry, vertex colour, lit)
    const mcDrawMod = d.createShaderModule({ code: COMMON + PROC + /* wgsl */ `
@group(0) @binding(0) var<uniform> U: Uniforms;
struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) nWorld: vec3f,
  @location(1) viewPos: vec3f,
  @location(2) color: vec3f,
  @location(3) worldPos: vec3f,
  @location(4) water: f32,
};
@vertex fn vs(@location(0) p: vec3f, @location(1) n: vec3f, @location(2) c: vec3f) -> VOut {
  var o: VOut;
  let vp = U.view * vec4f(p, 1.0);
  o.clip = U.proj * vp;
  o.nWorld = n;
  o.viewPos = vp.xyz;
  o.color = c;
  o.worldPos = p;
  // crude body-vs-water tag from colour (water is the blue group)
  o.water = select(0.0, 1.0, c.b > c.r + 0.15 && c.b > c.g);
  return o;
}
@fragment fn fs(in: VOut) -> @location(0) vec4f {
  let nW0 = normalize(in.nWorld);
  let s = triplanar(in.worldPos, nW0, in.color);
  let useTex = U.p2.x > 0.5 && in.water < 0.5; // mc has no rest grid → triplanar for both
  var albedo = select(in.color, s.rgb, useTex);
  let nW = select(nW0, bumpNormal(nW0, in.worldPos, s.a, 0.05), useTex);
  var n = normalize((U.view * vec4f(nW, 0.0)).xyz);
  let v = normalize(-in.viewPos);
  if (dot(n, v) < 0.0) { n = -n; }
  let l = normalize(-U.lightView.xyz);
  let diff = max(dot(n, l), 0.0);
  let h = normalize(l + v);
  let spec = pow(max(dot(n, h), 0.0), 40.0) * 0.5;
  let rim = pow(1.0 - max(dot(n, v), 0.0), 2.5) * 0.4;
  let ambient = 0.35 + 0.15 * n.y;
  let c = albedo * (ambient + diff * 0.85) + vec3f(spec) + vec3f(rim * 0.35);
  return vec4f(c, 1.0);
}
`});
    this.pipeMcDraw = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.uboBGL] }),
      vertex: {
        module: mcDrawMod, entryPoint: 'vs',
        buffers: [{
          arrayStride: 36,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32x3' },
            { shaderLocation: 2, offset: 24, format: 'float32x3' },
          ],
        }],
      },
      fragment: { module: mcDrawMod, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: this.depthFmt, depthWriteEnabled: true, depthCompare: 'less' },
    });

    this.mcVerts = d.createBuffer({
      size: MC_MAX_VERTS * 9 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX,
    });
    this.mcCounter = d.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
    });
    // mcBG (which references the per-sim density buffer) is built in
    // _buildSimBindGroups once attachSim has created densBuf.

    // ===================== raymarch (fullscreen) =====================
    const rmMod = d.createShaderModule({ code: COMMON + SAMPLE + PROC + FULLSCREEN_VS + /* wgsl */ `
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> dens: array<u32>;
@group(0) @binding(2) var<storage, read> restGrid: array<i32>;
@group(1) @binding(0) var scene: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;

// material-space rest position: trilinear restAccum / trilinear body density
fn cellRest(ix: i32, iy: i32, iz: i32) -> vec3f {
  let nx = i32(U.gridDim.x); let ny = i32(U.gridDim.y); let nz = i32(U.gridDim.z);
  if (ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz) { return vec3f(0.0); }
  let idx = (((iz * ny) + iy) * nx + ix) * 4;
  return vec3f(f32(restGrid[idx]), f32(restGrid[idx + 1]), f32(restGrid[idx + 2])) / 2048.0;
}
fn sampleRestPos(wp: vec3f) -> vec3f {
  let gi = (wp - U.gridMin.xyz) / U.gridMin.w;
  let b = floor(gi); let f = gi - b;
  let ix = i32(b.x); let iy = i32(b.y); let iz = i32(b.z);
  var ra = vec3f(0.0); var bd = 0.0;
  for (var c = 0; c < 8; c++) {
    let dx = c & 1; let dy = (c >> 1) & 1; let dz = (c >> 2) & 1;
    let tw = (select(1.0 - f.x, f.x, dx == 1)) * (select(1.0 - f.y, f.y, dy == 1)) * (select(1.0 - f.z, f.z, dz == 1));
    ra += tw * cellRest(ix + dx, iy + dy, iz + dz);
    let ch = cellCh(ix + dx, iy + dy, iz + dz);
    bd += tw * (ch.x + ch.y);
  }
  return ra / max(bd, 1e-3);
}

struct FOut { @location(0) color: vec4f, @builtin(frag_depth) depth: f32 };

fn rayDir(uv: vec2f) -> vec3f {
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let vp = U.invProj * vec4f(ndc, 0.0, 1.0);
  let viewPos = vp.xyz / vp.w;
  let world = (U.invView * vec4f(viewPos, 1.0)).xyz;
  return normalize(world - U.camPos.xyz);
}

@fragment fn fs(in: VOut) -> FOut {
  var out: FOut;
  out.color = vec4f(0.0);
  out.depth = 1.0;

  let ro = U.camPos.xyz;
  let rd = rayDir(in.uv);
  let voxel = U.gridMin.w;
  let gmax = U.gridMin.xyz + vec3f(U.gridDim.xyz - vec3u(1u)) * voxel;

  // ray vs grid AABB
  let inv = 1.0 / rd;
  let t0 = (U.gridMin.xyz - ro) * inv;
  let t1 = (gmax - ro) * inv;
  let tsmall = min(t0, t1);
  let tbig = max(t0, t1);
  var tn = max(max(tsmall.x, tsmall.y), max(tsmall.z, 0.0));
  let tf = min(min(tbig.x, tbig.y), tbig.z);
  if (tn >= tf) { discard; }

  let iso = U.p0.x;
  let hq = U.p1.w;                       // >0.5 in the supersampled HQ mode
  let step = voxel * select(0.6, 0.4, hq > 0.5);
  var t = tn;
  var hit = -1.0;
  var ta = tn;
  for (var s = 0; s < 768; s++) {
    if (t > tf) { break; }
    if (fieldAt(ro + rd * t) >= iso) { hit = t; break; }
    ta = t;
    t += step;
  }
  if (hit < 0.0) { discard; }

  // bisection refine between the last empty sample (ta) and the first hit (t):
  // gives a crisp iso crossing independent of the marching step size
  var a = ta; var b = hit;
  for (var k = 0; k < 6; k++) {
    let m = 0.5 * (a + b);
    if (fieldAt(ro + rd * m) >= iso) { b = m; } else { a = m; }
  }
  let wp = ro + rd * b;

  // silhouette/shape comes from the real field above; for the shading normal we
  // can widen the gradient stencil to wash out per-particle metaball lumps
  // without rounding the actual surface — sharp shape, smoother shading.
  let nrad = voxel * select(1.0, 2.5, hq > 0.5);
  var n = -normalize(gradAtH(wp, nrad) + vec3f(1e-6));
  let cinfo = colorAt(wp);
  let isWater = cinfo.a > 0.5;
  // body texturing: texMode 1 = triplanar (world space), 2 = material (rest
  // space → sticks to the body as it deforms). Computed unconditionally
  // (derivatives need uniform control flow) then selected.
  let useMat = abs(U.p2.x - 2.0) < 0.5 && !isWater;
  let coords = select(wp, sampleRestPos(wp), useMat);
  let s = triplanar(coords, n, cinfo.rgb);
  let useTex = U.p2.x > 0.5 && !isWater;
  var albedo = select(cinfo.rgb, s.rgb, useTex);
  n = select(n, bumpNormal(n, wp, s.a, 0.05), useTex);
  let v = -rd;
  let l = normalize(-U.lightView.xyz);
  let diff = max(dot(n, l), 0.0);
  let h = normalize(l + v);
  let spec = pow(max(dot(n, h), 0.0), 48.0);
  let fres = pow(1.0 - max(dot(n, v), 0.0), U.p1.z);

  var rgb: vec3f;
  if (isWater) {
    // water: screen-space refraction + fresnel reflection
    let uvOff = in.uv + n.xy * U.p1.y;
    let bg = textureSampleLevel(scene, samp, uvOff, 0.0).rgb;
    // crude thickness from how far the ray can keep marching inside
    var thick = 0.0;
    var tt = hit;
    for (var s = 0; s < 64; s++) {
      tt += step;
      if (tt > tf) { break; }
      if (fieldAt(ro + rd * tt) < iso) { break; }
      thick += step;
    }
    let absorb = exp(-U.p1.x * thick);
    let refr = mix(cinfo.rgb, bg, absorb * 0.85);
    let sky = vec3f(0.7, 0.82, 0.95);
    rgb = mix(refr, sky, fres * 0.7) + vec3f(spec) * 0.6;
  } else {
    let ambient = 0.35 + 0.15 * n.y;
    rgb = albedo * (ambient + diff * 0.85) + vec3f(spec) * 0.5 + vec3f(fres * 0.25);
  }

  // output depth so glass/overlays and env occlusion stay correct
  let clip = U.proj * (U.view * vec4f(wp, 1.0));
  out.depth = clip.z / clip.w;
  out.color = vec4f(rgb, 1.0);
  return out;
}
`});
    this.sceneTexBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });
    this.pipeRaymarch = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.densReadBGL, this.sceneTexBGL] }),
      vertex: { module: rmMod, entryPoint: 'fsv' },
      fragment: { module: rmMod, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: this.depthFmt, depthWriteEnabled: true, depthCompare: 'less' },
    });

    // ===================== blit (copy color, no depth) =====================
    const blitMod = d.createShaderModule({ code: FULLSCREEN_VS + /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@fragment fn fs(in: VOut) -> @location(0) vec4f {
  return textureSampleLevel(src, samp, in.uv, 0.0);
}
`});
    this.blitBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });
    this.pipeBlit = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.blitBGL] }),
      vertex: { module: blitMod, entryPoint: 'fsv' },
      fragment: { module: blitMod, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    });

    // ===================== downsample (SSAA resolve) =====================
    // 2x2 box filter from the supersampled colour texture to the swapchain
    const downMod = d.createShaderModule({ code: FULLSCREEN_VS + /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@fragment fn fs(in: VOut) -> @location(0) vec4f {
  let t = 1.0 / vec2f(textureDimensions(src));
  var c = vec4f(0.0);
  c += textureSampleLevel(src, samp, in.uv + vec2f(-0.5, -0.5) * t, 0.0);
  c += textureSampleLevel(src, samp, in.uv + vec2f( 0.5, -0.5) * t, 0.0);
  c += textureSampleLevel(src, samp, in.uv + vec2f(-0.5,  0.5) * t, 0.0);
  c += textureSampleLevel(src, samp, in.uv + vec2f( 0.5,  0.5) * t, 0.0);
  return c * 0.25;
}
`});
    this.pipeDown = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.blitBGL] }),
      vertex: { module: downMod, entryPoint: 'fsv' },
      fragment: { module: downMod, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    });

    this._buildSSFR();
    this._buildEmbed();
  }

  // ===================== embedded mesh (sim cage skinning) =====================
  // A detailed UV-mapped box bound to the cube lattice: each vertex follows the
  // trilinear blend of its 8 enclosing particles, so the real mesh (sharp edges,
  // real UVs → real textures) deforms exactly with the physics. Cells whose
  // lattice bonds have torn drop their triangles for a ragged cut edge.
  _buildEmbed() {
    const d = this.device;
    this.embedBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    this.embedMatBGL = d.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
    });
    const mod = d.createShaderModule({ code: COMMON + PROC + /* wgsl */ `
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> pos: array<vec4f>;
struct Mat { color: vec4f };
@group(1) @binding(0) var<uniform> mat: Mat;

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) nWorld: vec3f,
  @location(1) viewPos: vec3f,
  @location(2) uv: vec2f,
  @location(3) worldPos: vec3f,
};

// (F^-1)^T for normal transform; columns are the cofactors
fn normalMat(a: vec3f, b: vec3f, c: vec3f) -> mat3x3f {
  let r0 = cross(b, c); let r1 = cross(c, a); let r2 = cross(a, b);
  let det = dot(a, r0);
  return mat3x3f(r0, r1, r2) * (1.0 / det);
}

@vertex fn vs(
  @location(0) idxA: vec4u, @location(1) idxB: vec4u,
  @location(2) wA: vec4f, @location(3) wB: vec4f,
  @location(4) uv: vec2f, @location(5) restN: vec3f) -> VOut {
  var o: VOut;
  o.uv = uv;
  // barycentric binding to the 4 corners of the enclosing voxel (idxB/wB unused)
  let c0 = pos[idxA.x].xyz; let c1 = pos[idxA.y].xyz; let c2 = pos[idxA.z].xyz; let c3 = pos[idxA.w].xyz;
  let dp = c0*wA.x + c1*wA.y + c2*wA.z + c3*wA.w;
  o.worldPos = dp;
  let sp = U.p2.y;
  // tear detection: largest line stretch vs its rest (long 2a, short sqrt(3)a)
  let sl = max(length(c1-c0), length(c3-c2)) / (2.0 * sp);
  let ss = max(max(length(c2-c0), length(c3-c0)), max(length(c2-c1), length(c3-c1))) / (1.7320508 * sp);
  if (max(sl, ss) > U.p2.z) {
    o.clip = vec4f(2.0, 2.0, 2.0, 1.0);  // off-screen -> triangle dropped
    o.nWorld = vec3f(0.0, 1.0, 0.0); o.viewPos = vec3f(0.0);
    return o;
  }
  // restN holds Dm^T n, so the deformed normal is Ds^-T (Dm^T n) = F^-T n
  o.nWorld = normalize(normalMat(c1 - c0, c2 - c0, c3 - c0) * restN);
  let vp = U.view * vec4f(dp, 1.0);
  o.viewPos = vp.xyz;
  o.clip = U.proj * vp;
  return o;
}

@fragment fn fs(in: VOut) -> @location(0) vec4f {
  // compute texture unconditionally (dpdx/dpdy need uniform control flow), select
  let s = procSurf(in.uv, mat.color.rgb);
  let nW0 = normalize(in.nWorld);
  let textured = U.p2.x > 0.5;
  var albedo = select(mat.color.rgb, s.rgb, textured);
  var nW = select(nW0, bumpNormal(nW0, in.worldPos, s.a, 0.06), textured);
  var n = normalize((U.view * vec4f(nW, 0.0)).xyz);
  let v = normalize(-in.viewPos);
  if (dot(n, v) < 0.0) { n = -n; }
  let l = normalize(-U.lightView.xyz);
  let diff = max(dot(n, l), 0.0);
  let h = normalize(l + v);
  let spec = pow(max(dot(n, h), 0.0), 40.0) * 0.4;
  let rim = pow(1.0 - max(dot(n, v), 0.0), 2.5) * 0.3;
  let amb = 0.35 + 0.15 * n.y;
  return vec4f(albedo * (amb + diff * 0.85) + vec3f(spec) + vec3f(rim * 0.3), 1.0);
}
`});
    this.pipeEmbed = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.embedBGL, this.embedMatBGL] }),
      vertex: {
        module: mod, entryPoint: 'vs',
        buffers: [{
          arrayStride: 96,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'uint32x4' },
            { shaderLocation: 1, offset: 16, format: 'uint32x4' },
            { shaderLocation: 2, offset: 32, format: 'float32x4' },
            { shaderLocation: 3, offset: 48, format: 'float32x4' },
            { shaderLocation: 4, offset: 64, format: 'float32x2' },
            { shaderLocation: 5, offset: 72, format: 'float32x3' },
          ],
        }],
      },
      fragment: { module: mod, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: this.depthFmt, depthWriteEnabled: true, depthCompare: 'less' },
    });
  }

  // build the embedded render meshes + their voxel binding (per body): every
  // mesh vertex follows the 4 corners of its enclosing disphenoid through
  // barycentric weights (nearest voxel, extrapolated, for points just outside)
  _buildEmbedMeshes() {
    const d = this.device;
    const sim = this.sim;
    const a = sim.o.spacing;
    const { vertRest: VR, tetVerts: TV } = sim.host;
    if (this.embeds) for (const e of this.embeds) { e.vb.destroy(); e.matUB.destroy(); }
    this.embeds = [];
    const COLS = [COL_RED, COL_GREEN];
    const X = (v, k) => VR[v * 3 + k];
    for (const body of sim.bodies) {
      const { center, size, vox0, voxN } = body;
      // bucket voxel centroids on an a-sized grid
      const buckets = new Map();
      const bkey = (x, y, z) => `${x},${y},${z}`;
      for (let t = vox0; t < vox0 + voxN; t++) {
        let cx = 0, cy = 0, cz = 0;
        for (let k = 0; k < 4; k++) { const v = TV[t * 4 + k]; cx += X(v, 0); cy += X(v, 1); cz += X(v, 2); }
        const kk = bkey(Math.floor(cx / 4 / a), Math.floor(cy / 4 / a), Math.floor(cz / 4 / a));
        if (!buckets.has(kk)) buckets.set(kk, []);
        buckets.get(kk).push(t);
      }
      const bary = (t, p) => {
        const v = [0, 1, 2, 3].map((k) => TV[t * 4 + k]);
        const o = [X(v[0], 0), X(v[0], 1), X(v[0], 2)];
        const e = [1, 2, 3].map((k) => [X(v[k], 0) - o[0], X(v[k], 1) - o[1], X(v[k], 2) - o[2]]);
        const q = [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
        const cr = (u, w) => [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
        const dt = (u, w) => u[0] * w[0] + u[1] * w[1] + u[2] * w[2];
        const det = dt(e[0], cr(e[1], e[2]));
        const l1 = dt(q, cr(e[1], e[2])) / det, l2 = dt(e[0], cr(q, e[2])) / det, l3 = dt(e[0], cr(e[1], q)) / det;
        return { v, w: [1 - l1 - l2 - l3, l1, l2, l3], e };
      };
      const seg = Math.max(12, Math.min(48, body.cells * 4));
      const geo = new THREE.BoxGeometry(size, size, size, seg, seg, seg);
      geo.translate(center[0], center[1], center[2]);
      const g = geo.toNonIndexed();
      const Pp = g.attributes.position.array, Nn = g.attributes.normal.array, UV = g.attributes.uv.array;
      const count = Pp.length / 3;
      const stride = 24; // floats (96 bytes)
      const fbuf = new Float32Array(count * stride);
      const ubuf = new Uint32Array(fbuf.buffer);
      const memo = new Map(); // non-indexed geometry repeats each corner ~6x
      for (let i = 0; i < count; i++) {
        const p = [Pp[i * 3], Pp[i * 3 + 1], Pp[i * 3 + 2]];
        const pk = p.join();
        let best = memo.get(pk) ?? null, bestMin = best ? Infinity : -Infinity;
        const bx = Math.floor(p[0] / a), by = Math.floor(p[1] / a), bz = Math.floor(p[2] / a);
        for (let r = 1; r <= 3 && !best; r++)
          for (let dx = -r; dx <= r; dx++) for (let dy = -r; dy <= r; dy++) for (let dz = -r; dz <= r; dz++) {
            const list = buckets.get(bkey(bx + dx, by + dy, bz + dz));
            if (!list) continue;
            for (const t of list) {
              const b = bary(t, p);
              const m = Math.min(...b.w);
              if (m > bestMin) { bestMin = m; best = b; }
            }
          }
        memo.set(pk, best);
        const o = i * stride;
        if (best) {
          for (let k = 0; k < 4; k++) { ubuf[o + k] = best.v[k]; ubuf[o + 4 + k] = best.v[0]; fbuf[o + 8 + k] = best.w[k]; }
          // Dm^T n (Dm columns = rest edges from corner 0)
          const n = [Nn[i * 3], Nn[i * 3 + 1], Nn[i * 3 + 2]];
          for (let k = 0; k < 3; k++) {
            const e = best.e[k];
            fbuf[o + 18 + k] = e[0] * n[0] + e[1] * n[1] + e[2] * n[2];
          }
        }
        fbuf[o + 16] = UV[i * 2]; fbuf[o + 17] = UV[i * 2 + 1];
      }
      const vb = d.createBuffer({ size: fbuf.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(vb, 0, fbuf);
      const matUB = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(matUB, 0, new Float32Array([...(COLS[body.body] || COL_RED), 1]));
      this.embeds.push({
        vb, count, matUB,
        matBG: d.createBindGroup({ layout: this.embedMatBGL, entries: [{ binding: 0, resource: { buffer: matUB } }] }),
      });
    }
  }

  // ===================== SSFR (screen-space fluids) =====================
  _buildSSFR() {
    const d = this.device;
    // per-group material uniform: color(4) radius/water/_/_ (4) = 32 bytes
    this.ssfrMatBGL = d.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
    });
    // geometry pass binds: U + sim pos + flags
    this.ssfrGeomBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });

    // depth + albedo + linear viewZ from particle spheres
    const geomMod = d.createShaderModule({ code: COMMON + /* wgsl */ `
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> pos: array<vec4f>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
struct Mat { color: vec4f, info: vec4f }; // info: radius, water, base, _
@group(1) @binding(0) var<uniform> mat: Mat;

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) uv: vec2f,
  @location(1) viewCenter: vec3f,
};
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let i = u32(mat.info.z) + ii;
  var o: VOut;
  if ((flags[i] & 1u) == 0u) { o.clip = vec4f(0.0, 0.0, 2.0, 1.0); return o; }
  let corner = vec2f(f32(vi & 1u) * 2.0 - 1.0, f32(vi >> 1u) * 2.0 - 1.0);
  let vc = (U.view * vec4f(pos[i].xyz, 1.0)).xyz;
  let vp = vc + vec3f(corner * mat.info.x, 0.0);
  o.clip = U.proj * vec4f(vp, 1.0);
  o.uv = corner;
  o.viewCenter = vc;
  return o;
}
struct GOut {
  @location(0) albedo: vec4f,  // rgb + water flag
  @location(1) viewZ: vec4f,   // eye-space z (negative)
  @builtin(frag_depth) depth: f32,
};
@fragment fn fs(in: VOut) -> GOut {
  let r2 = dot(in.uv, in.uv);
  if (r2 > 1.0) { discard; }
  let nz = sqrt(1.0 - r2);
  let surf = in.viewCenter + vec3f(0.0, 0.0, mat.info.x * nz);
  let clip = U.proj * vec4f(surf, 1.0);
  var o: GOut;
  o.albedo = vec4f(mat.color.rgb, mat.info.y);
  o.viewZ = vec4f(surf.z, 0.0, 0.0, 1.0);
  o.depth = clip.z / clip.w;
  return o;
}
`});
    this.pipeSSFRGeom = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.ssfrGeomBGL, this.ssfrMatBGL] }),
      vertex: { module: geomMod, entryPoint: 'vs' },
      fragment: {
        module: geomMod, entryPoint: 'fs',
        targets: [{ format: 'rgba8unorm' }, { format: 'r32float' }],
      },
      primitive: { topology: 'triangle-strip' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    });

    // thickness: additive accumulation of water spheres
    const thickMod = d.createShaderModule({ code: COMMON + /* wgsl */ `
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> pos: array<vec4f>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
struct Mat { color: vec4f, info: vec4f }
@group(1) @binding(0) var<uniform> mat: Mat;
struct VOut { @builtin(position) clip: vec4f, @location(0) uv: vec2f, @location(1) tk: f32 };
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let i = u32(mat.info.z) + ii;
  var o: VOut;
  if ((flags[i] & 1u) == 0u || mat.info.y < 0.5) { o.clip = vec4f(0.0, 0.0, 2.0, 1.0); return o; }
  let corner = vec2f(f32(vi & 1u) * 2.0 - 1.0, f32(vi >> 1u) * 2.0 - 1.0);
  let vc = (U.view * vec4f(pos[i].xyz, 1.0)).xyz;
  let vp = vc + vec3f(corner * mat.info.x, 0.0);
  o.clip = U.proj * vec4f(vp, 1.0);
  o.uv = corner;
  o.tk = mat.info.x;
  return o;
}
@fragment fn fs(in: VOut) -> @location(0) vec4f {
  let r2 = dot(in.uv, in.uv);
  if (r2 > 1.0) { discard; }
  return vec4f(2.0 * in.tk * sqrt(1.0 - r2), 0.0, 0.0, 1.0);
}
`});
    const addBlend = {
      color: { srcFactor: 'one', dstFactor: 'one' },
      alpha: { srcFactor: 'one', dstFactor: 'one' },
    };
    this.pipeSSFRThick = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.ssfrGeomBGL, this.ssfrMatBGL] }),
      vertex: { module: thickMod, entryPoint: 'vs' },
      fragment: { module: thickMod, entryPoint: 'fs', targets: [{ format: 'r16float', blend: addBlend }] },
      primitive: { topology: 'triangle-strip' },
    });

    // separable bilateral blur of viewZ
    const blurMod = d.createShaderModule({ code: FULLSCREEN_VS + /* wgsl */ `
@group(0) @binding(0) var<uniform> dir: vec4f; // (dx, dy, 0, 0) in texels
@group(0) @binding(1) var src: texture_2d<f32>;
@fragment fn fs(in: VOut) -> @location(0) vec4f {
  let pc = vec2i(in.pos.xy);
  let center = textureLoad(src, pc, 0).r;
  if (center >= 0.0) { return vec4f(0.0, 0.0, 0.0, 1.0); } // background (z<0 for hits)
  var sum = 0.0;
  var wsum = 0.0;
  let R = 12;
  for (var k = -R; k <= R; k++) {
    let o = pc + vec2i(dir.xy) * k;
    let s = textureLoad(src, o, 0).r;
    if (s >= 0.0) { continue; }
    let sp = f32(k) / f32(R);
    let wsp = exp(-sp * sp * 4.0);             // spatial
    let dz = (s - center);
    let wr = exp(-dz * dz * 40.0);             // range (edge-preserving)
    let w = wsp * wr;
    sum += s * w;
    wsum += w;
  }
  let v = select(center, sum / wsum, wsum > 0.0);
  return vec4f(v, 0.0, 0.0, 1.0);
}
`});
    this.blurUB = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.blurUBv = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.blurBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      ],
    });
    this.pipeBlur = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.blurBGL] }),
      vertex: { module: blurMod, entryPoint: 'fsv' },
      fragment: { module: blurMod, entryPoint: 'fs', targets: [{ format: 'r32float' }] },
      primitive: { topology: 'triangle-list' },
    });

    // composite: reconstruct normal, shade water/body, sample scene for refraction
    const compMod = d.createShaderModule({ code: COMMON + PROC + FULLSCREEN_VS + /* wgsl */ `
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(1) @binding(0) var viewZTex: texture_2d<f32>;
@group(1) @binding(1) var albedoTex: texture_2d<f32>;
@group(1) @binding(2) var thickTex: texture_2d<f32>;
@group(1) @binding(3) var scene: texture_2d<f32>;
@group(1) @binding(4) var samp: sampler;

fn viewPosFrom(uv: vec2f, z: f32) -> vec3f {
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let vp = U.invProj * vec4f(ndc, 0.0, 1.0);
  var dir = vp.xyz / vp.w;
  dir = dir / dir.z; // ray with z = 1 at unit eye depth
  return dir * z;    // z is negative (eye space) -> correct position
}

struct FOut { @location(0) color: vec4f, @builtin(frag_depth) depth: f32 };

@fragment fn fs(in: VOut) -> FOut {
  let pc = vec2i(in.pos.xy);
  let z = textureLoad(viewZTex, pc, 0).r;
  if (z >= 0.0) { discard; }  // no particle here

  let texel = U.screen.zw;
  let zL = textureLoad(viewZTex, pc + vec2i(-1, 0), 0).r;
  let zR = textureLoad(viewZTex, pc + vec2i( 1, 0), 0).r;
  let zD = textureLoad(viewZTex, pc + vec2i(0, -1), 0).r;
  let zU = textureLoad(viewZTex, pc + vec2i(0,  1), 0).r;

  let P  = viewPosFrom(in.uv, z);
  // use the nearer finite difference on each axis (van der Laan) to avoid
  // smearing silhouettes
  var dx = viewPosFrom(in.uv + vec2f(texel.x, 0.0), zR) - P;
  let dxb = P - viewPosFrom(in.uv - vec2f(texel.x, 0.0), zL);
  if (zR >= 0.0 || abs(dxb.z) < abs(dx.z)) { dx = dxb; }
  var dy = viewPosFrom(in.uv + vec2f(0.0, texel.y), zU) - P;
  let dyb = P - viewPosFrom(in.uv - vec2f(0.0, texel.y), zD);
  if (zU >= 0.0 || abs(dyb.z) < abs(dy.z)) { dy = dyb; }

  var n = normalize(cross(dx, dy));
  if (n.z < 0.0) { n = -n; } // face the camera (eye space, looking down -z)

  let albedo = textureLoad(albedoTex, pc, 0);
  let isWater = albedo.a;
  let thick = textureLoad(thickTex, pc, 0).r;

  // triplanar procedural texture on the bodies (texMode 1); unconditional
  // compute (derivatives) then select
  let wp = (U.invView * vec4f(P, 1.0)).xyz;
  let wn0 = normalize((U.invView * vec4f(n, 0.0)).xyz);
  let s = triplanar(wp, wn0, albedo.rgb);
  let wnB = bumpNormal(wn0, wp, s.a, 0.05);
  let useTex = U.p2.x > 0.5 && isWater < 0.5; // ssfr has no rest grid → triplanar for both
  var alb = select(albedo.rgb, s.rgb, useTex);
  n = select(n, normalize((U.view * vec4f(wnB, 0.0)).xyz), useTex);

  let v = normalize(-P);
  let l = normalize(-U.lightView.xyz);
  let diff = max(dot(n, l), 0.0);
  let h = normalize(l + v);
  let spec = pow(max(dot(n, h), 0.0), 60.0);
  let fres = pow(1.0 - max(dot(n, v), 0.0), U.p1.z);

  var rgb: vec3f;
  if (isWater > 0.5) {
    let uvOff = in.uv + n.xy * U.p1.y;
    let bg = textureSampleLevel(scene, samp, uvOff, 0.0).rgb;
    let absorb = exp(-U.p1.x * thick);
    let refr = mix(albedo.rgb, bg, clamp(absorb, 0.0, 1.0) * 0.9);
    let sky = vec3f(0.7, 0.82, 0.95);
    rgb = mix(refr, sky, fres * 0.7) + vec3f(spec) * 0.7;
  } else {
    let ambient = 0.35 + 0.15 * n.y;
    rgb = alb * (ambient + diff * 0.85) + vec3f(spec) * 0.5 + vec3f(fres * 0.25);
  }

  let clip = U.proj * vec4f(P, 1.0);
  var o: FOut;
  o.color = vec4f(rgb, 1.0);
  o.depth = clip.z / clip.w;
  return o;
}
`});
    this.compBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });
    this.pipeComposite = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.uboBGL, this.compBGL] }),
      vertex: { module: compMod, entryPoint: 'fsv' },
      fragment: { module: compMod, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: this.depthFmt, depthWriteEnabled: true, depthCompare: 'less' },
    });
  }

  // bind groups that reference sim buffers / the density buffer (rebuilt on attach)
  _buildSimBindGroups() {
    const d = this.device;
    this.densAtomicBG = d.createBindGroup({
      layout: this.densAtomicBGL,
      entries: [
        { binding: 0, resource: { buffer: this.ub } },
        { binding: 1, resource: { buffer: this.densBuf } },
        { binding: 2, resource: { buffer: this.sim.buf.pos } },
        { binding: 3, resource: { buffer: this.sim.buf.flags } },
        { binding: 4, resource: { buffer: this.restBuf } },
        { binding: 5, resource: { buffer: this.restPosBuf } },
      ],
    });
    this.densReadBG = d.createBindGroup({
      layout: this.densReadBGL,
      entries: [
        { binding: 0, resource: { buffer: this.ub } },
        { binding: 1, resource: { buffer: this.densBuf } },
        { binding: 2, resource: { buffer: this.restBuf } },
      ],
    });
    this.mcBG = d.createBindGroup({
      layout: this.mcBGL,
      entries: [
        { binding: 0, resource: { buffer: this.ub } },
        { binding: 1, resource: { buffer: this.densBuf } },
        { binding: 2, resource: { buffer: this.triBuf } },
        { binding: 3, resource: { buffer: this.mcVerts } },
        { binding: 4, resource: { buffer: this.mcCounter } },
      ],
    });
    this.uboBG = d.createBindGroup({
      layout: this.uboBGL,
      entries: [{ binding: 0, resource: { buffer: this.ub } }],
    });
    this.embedBG = d.createBindGroup({
      layout: this.embedBGL,
      entries: [
        { binding: 0, resource: { buffer: this.ub } },
        { binding: 1, resource: { buffer: this.sim.buf.pos } },
      ],
    });
    this.ssfrGeomBG = d.createBindGroup({
      layout: this.ssfrGeomBGL,
      entries: [
        { binding: 0, resource: { buffer: this.ub } },
        { binding: 1, resource: { buffer: this.sim.buf.pos } },
        { binding: 2, resource: { buffer: this.sim.buf.flags } },
      ],
    });
    // per-group SSFR material uniforms
    if (this.ssfrMats) for (const m of this.ssfrMats) m.ub.destroy();
    this.ssfrMats = this.groups.map((g) => {
      const ub = d.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const data = new Float32Array(8);
      data.set([...g.color, 1], 0);
      data.set([g.radius, g.water, g.base, 0], 4);
      d.queue.writeBuffer(ub, 0, data);
      return {
        ub, count: g.count,
        bg: d.createBindGroup({ layout: this.ssfrMatBGL, entries: [{ binding: 0, resource: { buffer: ub } }] }),
      };
    });
  }

  resize(w, h, ss = 1) {
    const rw = w * ss, rh = h * ss;
    this.outW = w; this.outH = h; this.ss = ss;
    if (rw === this.w && rh === this.h && this.sceneTex) return;
    this.w = rw; this.h = rh;
    const d = this.device;
    const mk = (format, extra = 0) => d.createTexture({
      size: [rw, rh], format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | extra,
    });
    for (const t of ['sceneTex', 'sceneTex2', 'depthTex', 'ssfrAlbedo', 'ssfrViewZ', 'ssfrTmp', 'ssfrBlur', 'ssfrThick', 'ssfrDepth']) {
      if (this[t]) this[t].destroy();
    }
    this.sceneTex = mk(this.format);
    this.sceneTex2 = mk(this.format);
    this.depthTex = mk(this.depthFmt);
    this.ssfrAlbedo = mk('rgba8unorm');
    this.ssfrViewZ = mk('r32float');
    this.ssfrTmp = mk('r32float');
    this.ssfrBlur = mk('r32float');
    this.ssfrThick = mk('r16float');
    this.ssfrDepth = mk('depth32float');

    this.blitBG = d.createBindGroup({
      layout: this.blitBGL,
      entries: [
        { binding: 0, resource: this.sceneTex.createView() },
        { binding: 1, resource: this.sampler },
      ],
    });
    this.raymarchSceneBG = d.createBindGroup({
      layout: this.sceneTexBGL,
      entries: [
        { binding: 0, resource: this.sceneTex.createView() },
        { binding: 1, resource: this.sampler },
      ],
    });
    this.blurBGh = d.createBindGroup({
      layout: this.blurBGL,
      entries: [
        { binding: 0, resource: { buffer: this.blurUB } },
        { binding: 1, resource: this.ssfrViewZ.createView() },
      ],
    });
    this.blurBGv = d.createBindGroup({
      layout: this.blurBGL,
      entries: [
        { binding: 0, resource: { buffer: this.blurUBv } },
        { binding: 1, resource: this.ssfrTmp.createView() },
      ],
    });
    this.compBG = d.createBindGroup({
      layout: this.compBGL,
      entries: [
        { binding: 0, resource: this.ssfrBlur.createView() },
        { binding: 1, resource: this.ssfrAlbedo.createView() },
        { binding: 2, resource: this.ssfrThick.createView() },
        { binding: 3, resource: this.sceneTex.createView() },
        { binding: 4, resource: this.sampler },
      ],
    });
    d.queue.writeBuffer(this.blurUB, 0, new Float32Array([1, 0, 0, 0]));
    d.queue.writeBuffer(this.blurUBv, 0, new Float32Array([0, 1, 0, 0]));
  }

  // upload per-frame uniforms. camera = THREE.PerspectiveCamera (matrices ready)
  updateUniforms(camera, lightDir, iso, opts = {}) {
    const { hq = false, texMode = 0, waterOnly = false } = opts;
    const f = this.uf, u = this.uu;
    const view = camera.matrixWorldInverse;
    const invView = camera.matrixWorld;
    const proj = camera.projectionMatrix;
    // three matrices are column-major Float32Array(16) — copy straight in
    f.set(view.elements, 0);
    f.set(proj.elements, 16);
    f.set(invertMat4(proj.elements), 32);
    f.set(invView.elements, 48);
    f.set([camera.position.x, camera.position.y, camera.position.z, 0], 64);
    const lv = lightDir.clone().transformDirection(view);
    f.set([lv.x, lv.y, lv.z, 0], 68);
    f.set([this.w, this.h, 1 / this.w, 1 / this.h], 72);
    f.set([this.gridMin[0], this.gridMin[1], this.gridMin[2], this.voxel], 76);
    u.set([this.dim[0], this.dim[1], this.dim[2], 0], 80);
    u.set([this.sim.N1, this.sim.SOLID_N, this.sim.TOTAL, this.sim.BALL0], 84);
    f.set([iso, this.waterR, this.bodyR, 0], 88);
    f.set([2.5, 0.05, 3.0, hq ? 1 : 0], 92); // absorb, refractScale, fresnelPow, hqFlag
    f.set([texMode, this.sim.o.spacing, (this.sim.o.params.tearGap || 1.3) * 1.25, waterOnly ? 1 : 0], 96);
    this.device.queue.writeBuffer(this.ub, 0, this.uf);
  }

  // ---- shared density build (mc + raymarch) ----
  buildDensity(enc) {
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipeClear);
    pass.setBindGroup(0, this.densAtomicBG);
    pass.dispatchWorkgroups(Math.min(65535, Math.ceil((this.nVox * 4) / 64)));
    pass.setPipeline(this.pipeSplat);
    pass.setBindGroup(0, this.densAtomicBG);
    pass.dispatchWorkgroups(Math.ceil(this.sim.TOTAL / 64));
    pass.end();
  }

  // ---- main entry: render a skinned frame to swapchainView ----
  // drawEnv(pass) draws opaque environment; drawOverlays(pass) draws glass+trail.
  render(enc, swapchainView, mode, drawEnv, drawOverlays) {
    const clearScene = { r: 0.063, g: 0.078, b: 0.11, a: 1 };

    // ----- pass A: environment into sceneTex + depth -----
    const envPass = enc.beginRenderPass({
      colorAttachments: [{ view: this.sceneTex.createView(), clearValue: clearScene, loadOp: 'clear', storeOp: 'store' }],
      depthStencilAttachment: { view: this.depthTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
    });
    drawEnv(envPass);
    envPass.end();

    let colorTex = this.sceneTex;

    if (mode === 'mc') {
      this.buildDensity(enc);
      // clear + march
      let cp = enc.beginComputePass();
      cp.setPipeline(this.pipeMcClear);
      cp.setBindGroup(0, this.mcBG);
      cp.dispatchWorkgroups(1);
      cp.setPipeline(this.pipeMcMarch);
      cp.setBindGroup(0, this.mcBG);
      cp.dispatchWorkgroups(
        Math.ceil((this.dim[0] - 1) / 4),
        Math.ceil((this.dim[1] - 1) / 4),
        Math.ceil((this.dim[2] - 1) / 4));
      cp.end();
      // draw the extracted mesh into sceneTex (load env color + depth)
      const mcPass = enc.beginRenderPass({
        colorAttachments: [{ view: this.sceneTex.createView(), loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: this.depthTex.createView(), depthLoadOp: 'load', depthStoreOp: 'store' },
      });
      mcPass.setPipeline(this.pipeMcDraw);
      mcPass.setBindGroup(0, this.uboBG);
      mcPass.setVertexBuffer(0, this.mcVerts);
      mcPass.drawIndirect(this.mcCounter, 0);
      mcPass.end();
    } else if (mode === 'embed') {
      // water surface via marching cubes (water-only field, set in uniforms),
      // then the embedded body meshes — both real geometry into sceneTex+depth
      this.buildDensity(enc);
      let cp = enc.beginComputePass();
      cp.setPipeline(this.pipeMcClear);
      cp.setBindGroup(0, this.mcBG);
      cp.dispatchWorkgroups(1);
      cp.setPipeline(this.pipeMcMarch);
      cp.setBindGroup(0, this.mcBG);
      cp.dispatchWorkgroups(
        Math.ceil((this.dim[0] - 1) / 4),
        Math.ceil((this.dim[1] - 1) / 4),
        Math.ceil((this.dim[2] - 1) / 4));
      cp.end();
      const ePass = enc.beginRenderPass({
        colorAttachments: [{ view: this.sceneTex.createView(), loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: this.depthTex.createView(), depthLoadOp: 'load', depthStoreOp: 'store' },
      });
      ePass.setPipeline(this.pipeMcDraw);
      ePass.setBindGroup(0, this.uboBG);
      ePass.setVertexBuffer(0, this.mcVerts);
      ePass.drawIndirect(this.mcCounter, 0);
      ePass.setPipeline(this.pipeEmbed);
      ePass.setBindGroup(0, this.embedBG);
      for (const e of this.embeds) {
        ePass.setBindGroup(1, e.matBG);
        ePass.setVertexBuffer(0, e.vb);
        ePass.draw(e.count);
      }
      ePass.end();
    } else if (mode === 'raymarch' || mode === 'raymarchHQ') {
      this.buildDensity(enc);
      // copy env so passthrough pixels keep the background
      this._blit(enc, this.sceneTex2);
      const rmPass = enc.beginRenderPass({
        colorAttachments: [{ view: this.sceneTex2.createView(), loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: this.depthTex.createView(), depthLoadOp: 'load', depthStoreOp: 'store' },
      });
      rmPass.setPipeline(this.pipeRaymarch);
      rmPass.setBindGroup(0, this.densReadBG);
      rmPass.setBindGroup(1, this.raymarchSceneBG);
      rmPass.draw(3);
      rmPass.end();
      colorTex = this.sceneTex2;
    } else if (mode === 'ssfr') {
      this._ssfr(enc);
      colorTex = this.sceneTex2;
    }

    // ----- overlays (glass + trail) into the final colour tex -----
    const ovPass = enc.beginRenderPass({
      colorAttachments: [{ view: colorTex.createView(), loadOp: 'load', storeOp: 'store' }],
      depthStencilAttachment: { view: this.depthTex.createView(), depthLoadOp: 'load', depthStoreOp: 'store' },
    });
    drawOverlays(ovPass);
    ovPass.end();

    // ----- present (downsample if supersampled) -----
    this.lastColorTex = colorTex;
    if (this.ss > 1) this._downsampleTo(enc, colorTex, swapchainView);
    else this._blitTo(enc, colorTex, swapchainView);
  }

  _downsampleTo(enc, src, dstView) {
    const bg = this.device.createBindGroup({
      layout: this.blitBGL,
      entries: [
        { binding: 0, resource: src.createView() },
        { binding: 1, resource: this.sampler },
      ],
    });
    const p = enc.beginRenderPass({
      colorAttachments: [{ view: dstView, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store' }],
    });
    p.setPipeline(this.pipeDown);
    p.setBindGroup(0, bg);
    p.draw(3);
    p.end();
  }

  // debug: read back the final composited colour texture and report the mean /
  // max brightness over the right portion of the frame (where the scene is).
  async debugReadColor() {
    if (!this.lastColorTex) return { err: 'no frame yet' };
    const d = this.device, w = this.w, h = this.h;
    const bpr = Math.ceil((w * 4) / 256) * 256;
    const buf = d.createBuffer({ size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = d.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: this.lastColorTex }, { buffer: buf, bytesPerRow: bpr }, { width: w, height: h });
    d.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const data = new Uint8Array(buf.getMappedRange());
    let sum = 0, n = 0, max = 0;
    const x0 = Math.floor(w * 0.35);
    for (let y = 0; y < h; y++) {
      for (let x = x0; x < w; x++) {
        const i = y * bpr + x * 4;
        const v = (data[i] + data[i + 1] + data[i + 2]) / 3;
        sum += v; max = Math.max(max, v); n++;
      }
    }
    buf.unmap(); buf.destroy();
    return { mean: +(sum / n).toFixed(1), max };
  }

  // debug: reconstruct the final colour texture as a PNG data URL (works even
  // where the browser won't composite a WebGPU canvas into a screenshot)
  async debugReadPNG() {
    if (!this.lastColorTex) return null;
    const d = this.device, w = this.w, h = this.h;
    const bpr = Math.ceil((w * 4) / 256) * 256;
    const buf = d.createBuffer({ size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = d.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: this.lastColorTex }, { buffer: buf, bytesPerRow: bpr }, { width: w, height: h });
    d.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(buf.getMappedRange());
    const cv = document.createElement('canvas');
    cv.width = w; cv.height = h;
    const ctx = cv.getContext('2d');
    const img = ctx.createImageData(w, h);
    const bgra = this.format.startsWith('bgra');
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const s = y * bpr + x * 4, dst = (y * w + x) * 4;
        if (bgra) { img.data[dst] = src[s + 2]; img.data[dst + 1] = src[s + 1]; img.data[dst + 2] = src[s]; }
        else { img.data[dst] = src[s]; img.data[dst + 1] = src[s + 1]; img.data[dst + 2] = src[s + 2]; }
        img.data[dst + 3] = 255;
      }
    }
    buf.unmap(); buf.destroy();
    ctx.putImageData(img, 0, 0);
    return cv.toDataURL('image/png');
  }

  _ssfr(enc) {
    // geometry: depth + albedo + viewZ
    const gPass = enc.beginRenderPass({
      colorAttachments: [
        { view: this.ssfrAlbedo.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
        { view: this.ssfrViewZ.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
      ],
      depthStencilAttachment: { view: this.ssfrDepth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
    });
    gPass.setPipeline(this.pipeSSFRGeom);
    gPass.setBindGroup(0, this.ssfrGeomBG);
    for (const m of this.ssfrMats) {
      if (m.count === 0) continue;
      gPass.setBindGroup(1, m.bg);
      gPass.draw(4, m.count);
    }
    gPass.end();

    // thickness (water only, additive)
    const tPass = enc.beginRenderPass({
      colorAttachments: [{ view: this.ssfrThick.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }],
    });
    tPass.setPipeline(this.pipeSSFRThick);
    tPass.setBindGroup(0, this.ssfrGeomBG);
    for (const m of this.ssfrMats) {
      if (m.count === 0) continue;
      tPass.setBindGroup(1, m.bg);
      tPass.draw(4, m.count);
    }
    tPass.end();

    // bilateral blur: viewZ -> tmp (H) -> blur (V)
    const blurPass = (dst, bg) => {
      const p = enc.beginRenderPass({
        colorAttachments: [{ view: dst.createView(), loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store' }],
      });
      p.setPipeline(this.pipeBlur);
      p.setBindGroup(0, bg);
      p.draw(3);
      p.end();
    };
    blurPass(this.ssfrTmp, this.blurBGh);
    blurPass(this.ssfrBlur, this.blurBGv);

    // copy env then composite the surface over it
    this._blit(enc, this.sceneTex2);
    const cPass = enc.beginRenderPass({
      colorAttachments: [{ view: this.sceneTex2.createView(), loadOp: 'load', storeOp: 'store' }],
      depthStencilAttachment: { view: this.depthTex.createView(), depthLoadOp: 'load', depthStoreOp: 'store' },
    });
    cPass.setPipeline(this.pipeComposite);
    cPass.setBindGroup(0, this.uboBG);
    cPass.setBindGroup(1, this.compBG);
    cPass.draw(3);
    cPass.end();
  }

  // copy sceneTex -> dst (color only)
  _blit(enc, dst) {
    const p = enc.beginRenderPass({
      colorAttachments: [{ view: dst.createView(), loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store' }],
    });
    p.setPipeline(this.pipeBlit);
    p.setBindGroup(0, this.blitBG);
    p.draw(3);
    p.end();
  }

  // present src -> swapchain
  _blitTo(enc, src, dstView) {
    const bg = src === this.sceneTex ? this.blitBG : this.device.createBindGroup({
      layout: this.blitBGL,
      entries: [
        { binding: 0, resource: src.createView() },
        { binding: 1, resource: this.sampler },
      ],
    });
    const p = enc.beginRenderPass({
      colorAttachments: [{ view: dstView, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store' }],
    });
    p.setPipeline(this.pipeBlit);
    p.setBindGroup(0, bg);
    p.draw(3);
    p.end();
  }
}

// 4x4 matrix inverse (column-major, matches three.js element order)
function invertMat4(m) {
  const inv = new Float32Array(16);
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];
  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10, b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30, b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return inv;
  det = 1.0 / det;
  inv[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  inv[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  inv[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  inv[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  inv[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  inv[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  inv[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  inv[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  inv[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  inv[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  inv[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  inv[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  inv[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  inv[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  inv[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  inv[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return inv;
}
