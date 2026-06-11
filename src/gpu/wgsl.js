// All WGSL kernels for the unified particle sim.
// Gather-only formulation: every kernel writes only to its own particle's
// slot (or a private delta), so no float atomics are needed anywhere.

export const makeShaderSource = (C) => /* wgsl */ `

// ---- constants baked at pipeline creation ----
const SOLID_N: u32 = ${C.SOLID_N}u;
const MAX_FLUID: u32 = ${C.MAX_FLUID}u;
const TOTAL: u32 = ${C.TOTAL}u;
const TABLE: u32 = ${C.TABLE}u;          // hash table size (pow2)
const NUM_CONS: u32 = ${C.NUM_CONS}u;
const SCAN_BLOCKS: u32 = ${C.SCAN_BLOCKS}u; // TABLE / 256

// scratch slots at the tail of the gridA buffer (after counts|cursor|entries)
const SCRATCH: u32 = TABLE + TABLE + TOTAL;
const S_PICK: u32 = SCRATCH;       // packed atomicMin pick result
const S_COUNT: u32 = SCRATCH + 1u; // active fluid counter

const F_ACTIVE: u32 = 1u;
const F_GRABBED: u32 = 2u;

struct Params {
  h: f32,               // substep dt
  gravity: f32,
  solidRadius: f32,
  fluidRadius: f32,

  cellSize: f32,
  fluidH: f32,
  restDensity: f32,
  stiffness: f32,

  nearStiffness: f32,
  viscosity: f32,
  compliance: f32,
  omega: f32,           // jacobi relaxation for solid constraints

  plasticThreshold: f32,
  plasticRate: f32,
  breakStrain: f32,
  maxSpeed: f32,

  wallX: f32,
  wallZ: f32,
  wallH: f32,
  wallT: f32,           // half thickness

  cullR: f32,
  dampSolid: f32,
  dampFluid: f32,
  frictionSolid: f32,

  frictionFluid: f32,
  grabK: f32,
  emitCount: u32,
  emitStart: u32,

  emitPos: vec4f,       // xyz + speed in w
  emitDir: vec4f,       // xyz + phase in w
  emitSide: vec4f,
  emitUp: vec4f,

  grabDelta: vec4f,     // per-substep kinematic move of grabbed particles
  cutO: vec4f,          // knife wedge: camera origin
  cutD1: vec4f,
  cutD2: vec4f,

  pickO: vec4f,         // pick ray origin, w = max perp dist
  pickD: vec4f,         // pick ray dir
  grabC: vec4f,         // grab-select sphere center, w = radius
}

@group(0) @binding(0) var<storage, read_write> pos: array<vec4f>;     // xyz + invMass
@group(0) @binding(1) var<storage, read_write> prev: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> flags: array<u32>;
@group(0) @binding(4) var<storage, read_write> delta: array<vec4f>;   // xyz + count
@group(0) @binding(5) var<storage, read_write> density: array<vec2f>;
@group(0) @binding(6) var<storage, read_write> cons: array<vec4u>;    // a, b, rest(f32 bits), alive
@group(0) @binding(7) var<storage, read_write> adj: array<u32>;       // [SOLID_N+1 offsets][constraint ids]
@group(0) @binding(8) var<storage, read_write> gridA: array<atomic<u32>>; // counts | cursor | entries | scratch
@group(0) @binding(9) var<storage, read_write> gridS: array<u32>;     // scanned [TABLE+1] | block partials
@group(0) @binding(10) var<uniform> P: Params;

fn effInvMass(i: u32) -> f32 {
  if ((flags[i] & F_GRABBED) != 0u) { return 0.0; }
  return pos[i].w;
}

fn cellHash(c: vec3i) -> u32 {
  let h = (u32(c.x) * 92837111u) ^ (u32(c.y) * 689287499u) ^ (u32(c.z) * 283923481u);
  return h & (TABLE - 1u);
}

fn posHash(p: vec3f) -> u32 {
  return cellHash(vec3i(floor(p / P.cellSize)));
}

// ============ integrate ============
@compute @workgroup_size(256)
fn integrate(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= TOTAL || (flags[i] & F_ACTIVE) == 0u) { return; }
  var v = vel[i].xyz;
  if ((flags[i] & F_GRABBED) == 0u) {
    v.y += P.gravity * P.h;
  } else {
    v = vec3f(0.0);
  }
  let p = pos[i].xyz;
  prev[i] = vec4f(p, 0.0);
  var np = p + v * P.h;
  if ((flags[i] & F_GRABBED) != 0u) { np = p + P.grabDelta.xyz; }
  pos[i] = vec4f(np, pos[i].w);
  vel[i] = vec4f(v, 0.0);
}

// ============ grid build ============
@compute @workgroup_size(256)
fn gridClear(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i < TABLE) {
    atomicStore(&gridA[i], 0u);          // counts
    atomicStore(&gridA[TABLE + i], 0u);  // cursor
  }
  if (i < 8u) { atomicStore(&gridA[SCRATCH + i], select(0u, 0xffffffffu, i == 0u)); }
}

@compute @workgroup_size(256)
fn gridCount(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= TOTAL || (flags[i] & F_ACTIVE) == 0u) { return; }
  atomicAdd(&gridA[posHash(pos[i].xyz)], 1u);
}

// hierarchical exclusive scan of counts -> gridS[0..TABLE], gridS[TABLE]=total
var<workgroup> wgScan: array<u32, 256>;

@compute @workgroup_size(256)
fn scanBlocks(@builtin(global_invocation_id) g: vec3u,
              @builtin(local_invocation_id) l: vec3u,
              @builtin(workgroup_id) w: vec3u) {
  let i = g.x;
  var v = 0u;
  if (i < TABLE) { v = atomicLoad(&gridA[i]); }
  wgScan[l.x] = v;
  workgroupBarrier();
  // inclusive Hillis-Steele
  for (var s = 1u; s < 256u; s = s << 1u) {
    var t = 0u;
    if (l.x >= s) { t = wgScan[l.x - s]; }
    workgroupBarrier();
    wgScan[l.x] += t;
    workgroupBarrier();
  }
  if (i < TABLE) { gridS[i] = wgScan[l.x] - v; } // exclusive
  if (l.x == 255u) { gridS[TABLE + 1u + w.x] = wgScan[255]; } // block total
}

@compute @workgroup_size(256)
fn scanPartials(@builtin(local_invocation_id) l: vec3u) {
  // scan SCAN_BLOCKS block totals with one workgroup (serial chunks)
  var carry = 0u;
  for (var base = 0u; base < SCAN_BLOCKS; base += 256u) {
    let idx = base + l.x;
    var v = 0u;
    if (idx < SCAN_BLOCKS) { v = gridS[TABLE + 1u + idx]; }
    wgScan[l.x] = v;
    workgroupBarrier();
    for (var s = 1u; s < 256u; s = s << 1u) {
      var t = 0u;
      if (l.x >= s) { t = wgScan[l.x - s]; }
      workgroupBarrier();
      wgScan[l.x] += t;
      workgroupBarrier();
    }
    if (idx < SCAN_BLOCKS) { gridS[TABLE + 1u + idx] = wgScan[l.x] - v + carry; }
    workgroupBarrier();
    carry += wgScan[255];
    workgroupBarrier();
  }
}

@compute @workgroup_size(256)
fn scanApply(@builtin(global_invocation_id) g: vec3u, @builtin(workgroup_id) w: vec3u) {
  let i = g.x;
  if (i < TABLE) { gridS[i] += gridS[TABLE + 1u + w.x]; }
  if (i == 0u) {
    // total = last block offset + nothing else needed; store in gridS[TABLE]
    gridS[TABLE] = 0u; // unused; ranges come from neighbor diff below
  }
}

@compute @workgroup_size(256)
fn gridScatter(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= TOTAL || (flags[i] & F_ACTIVE) == 0u) { return; }
  let h = posHash(pos[i].xyz);
  let slot = gridS[h] + atomicAdd(&gridA[TABLE + h], 1u);
  atomicStore(&gridA[TABLE + TABLE + slot], i);
}

// cell range helper: start = gridS[h], end = gridS[h] + counts[h]
fn cellRange(h: u32) -> vec2u {
  let s = gridS[h];
  return vec2u(s, s + atomicLoad(&gridA[h]));
}

// ============ fluid: double-density relaxation ============
@compute @workgroup_size(256)
fn fluidDensity(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + SOLID_N;
  if (i >= TOTAL || (flags[i] & F_ACTIVE) == 0u) { return; }
  let pi = pos[i].xyz;
  let H = P.fluidH;
  let H2 = H * H;
  let c0 = vec3i(floor(pi / P.cellSize));
  var rho = 0.0;
  var rhoNear = 0.0;
  for (var dx = -1; dx <= 1; dx++) {
  for (var dy = -1; dy <= 1; dy++) {
  for (var dz = -1; dz <= 1; dz++) {
    let r = cellRange(cellHash(c0 + vec3i(dx, dy, dz)));
    for (var e = r.x; e < r.y; e++) {
      let j = atomicLoad(&gridA[TABLE + TABLE + e]);
      if (j == i || j < SOLID_N) { continue; }
      let d2 = dot(pos[j].xyz - pi, pos[j].xyz - pi);
      if (d2 >= H2) { continue; }
      let q = 1.0 - sqrt(d2) / H;
      rho += q * q;
      rhoNear += q * q * q;
    }
  }}}
  density[i] = vec2f(rho, rhoNear);
}

@compute @workgroup_size(256)
fn fluidDisp(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + SOLID_N;
  if (i >= TOTAL || (flags[i] & F_ACTIVE) == 0u) { return; }
  let pi = pos[i].xyz;
  let H = P.fluidH;
  let H2 = H * H;
  let maxPair = 0.05 * H;
  let maxTotal = 0.35 * H;
  let Pi = P.stiffness * (density[i].x - P.restDensity);
  let PNi = P.nearStiffness * density[i].y;
  let c0 = vec3i(floor(pi / P.cellSize));
  var disp = vec3f(0.0);
  for (var dx = -1; dx <= 1; dx++) {
  for (var dy = -1; dy <= 1; dy++) {
  for (var dz = -1; dz <= 1; dz++) {
    let r = cellRange(cellHash(c0 + vec3i(dx, dy, dz)));
    for (var e = r.x; e < r.y; e++) {
      let j = atomicLoad(&gridA[TABLE + TABLE + e]);
      if (j == i || j < SOLID_N) { continue; }
      let dd = pos[j].xyz - pi;
      let d2 = dot(dd, dd);
      if (d2 >= H2 || d2 < 1e-12) { continue; }
      let d = sqrt(d2);
      let q = 1.0 - d / H;
      let Pj = P.stiffness * (density[j].x - P.restDensity);
      let PNj = P.nearStiffness * density[j].y;
      var D = 0.25 * ((Pi + Pj) * q + (PNi + PNj) * q * q);
      D = clamp(D, -maxPair, maxPair);
      disp -= dd * (D / d);
    }
  }}}
  let s2 = dot(disp, disp);
  if (s2 > maxTotal * maxTotal) { disp *= maxTotal / sqrt(s2); }
  delta[i] = vec4f(disp, 1.0);
}

@compute @workgroup_size(256)
fn applyDeltaFluid(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + SOLID_N;
  if (i >= TOTAL || (flags[i] & F_ACTIVE) == 0u) { return; }
  pos[i] = vec4f(pos[i].xyz + delta[i].xyz, pos[i].w);
}

// ============ solid: XPBD jacobi over adjacency ============
@compute @workgroup_size(256)
fn solidSolve(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= SOLID_N || (flags[i] & F_ACTIVE) == 0u) { return; }
  let pi = pos[i].xyz;
  let wi = effInvMass(i);
  let alpha = P.compliance / (P.h * P.h);
  var corr = vec3f(0.0);
  var cnt = 0.0;
  let start = adj[i];
  let end = adj[i + 1u];
  for (var k = start; k < end; k++) {
    let ci = adj[SOLID_N + 1u + k];
    let c = cons[ci];
    if (c.w == 0u) { continue; }
    let other = select(c.x, c.y, c.x == i);
    let rest = bitcast<f32>(c.z);
    let dd = pi - pos[other].xyz;
    let d = length(dd);
    if (d < 1e-9) { continue; }
    // tear on overstretch (both endpoints compute the same verdict)
    if (d > rest * P.breakStrain) { cons[ci].w = 0u; continue; }
    let wo = effInvMass(other);
    let wsum = wi + wo;
    if (wsum == 0.0) { continue; }
    let C = d - rest;
    let dl = -C / (wsum + alpha);
    corr += dd * (dl * wi / d);
    cnt += 1.0;
    // plasticity: permanent dents (same value written from both ends)
    let strain = C / rest;
    if (abs(strain) > P.plasticThreshold) {
      cons[ci].z = bitcast<u32>(rest + C * P.plasticRate);
    }
  }
  delta[i] = vec4f(corr, cnt);
}

@compute @workgroup_size(256)
fn applyDeltaSolid(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= SOLID_N || (flags[i] & F_ACTIVE) == 0u) { return; }
  let d = delta[i];
  if (d.w > 0.0) {
    pos[i] = vec4f(pos[i].xyz + d.xyz * (P.omega / d.w), pos[i].w);
  }
}

// ============ contacts: solid-solid (non-bonded) + solid-fluid ============
fn isBonded(i: u32, j: u32) -> bool {
  let start = adj[i];
  let end = adj[i + 1u];
  for (var k = start; k < end; k++) {
    let c = cons[adj[SOLID_N + 1u + k]];
    if (c.w != 0u && (c.x == j || c.y == j)) { return true; }
  }
  return false;
}

@compute @workgroup_size(256)
fn contacts(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= TOTAL || (flags[i] & F_ACTIVE) == 0u) { return; }
  let iSolid = i < SOLID_N;
  let pi = pos[i].xyz;
  let wi = effInvMass(i);
  let ri = select(P.fluidRadius, P.solidRadius, iSolid);
  let c0 = vec3i(floor(pi / P.cellSize));
  var corr = vec3f(0.0);
  var cnt = 0.0;
  for (var dx = -1; dx <= 1; dx++) {
  for (var dy = -1; dy <= 1; dy++) {
  for (var dz = -1; dz <= 1; dz++) {
    let r = cellRange(cellHash(c0 + vec3i(dx, dy, dz)));
    for (var e = r.x; e < r.y; e++) {
      let j = atomicLoad(&gridA[TABLE + TABLE + e]);
      if (j == i) { continue; }
      let jSolid = j < SOLID_N;
      if (!iSolid && !jSolid) { continue; } // fluid-fluid handled by DDR
      let dd = pi - pos[j].xyz;
      let rj = select(P.fluidRadius, P.solidRadius, jSolid);
      let rsum = ri + rj;
      let d2 = dot(dd, dd);
      if (d2 >= rsum * rsum || d2 < 1e-12) { continue; }
      if (iSolid && jSolid && isBonded(i, j)) { continue; }
      let wj = effInvMass(j);
      let wsum = wi + wj;
      if (wsum == 0.0) { continue; }
      let d = sqrt(d2);
      corr += dd * ((rsum - d) / d * wi / wsum);
      cnt += 1.0;
    }
  }}}
  delta[i] = vec4f(corr, cnt);
}

@compute @workgroup_size(256)
fn applyDeltaAll(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= TOTAL || (flags[i] & F_ACTIVE) == 0u) { return; }
  let d = delta[i];
  if (d.w > 0.0) {
    pos[i] = vec4f(pos[i].xyz + d.xyz / max(d.w, 1.0), pos[i].w);
  }
}

// ============ bounds ============
@compute @workgroup_size(256)
fn bounds(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= TOTAL || (flags[i] & F_ACTIVE) == 0u) { return; }
  let iSolid = i < SOLID_N;
  var p = pos[i].xyz;
  let pv = prev[i].xyz;
  let r = select(P.fluidRadius, P.solidRadius, iSolid);

  if (p.y < r) {
    p.y = r;
    let mu = select(P.frictionFluid, P.frictionSolid, iSolid);
    p.x -= (p.x - pv.x) * mu;
    p.z -= (p.z - pv.z) * mu;
  }

  if (p.y < P.wallH + r) {
    let lim = r + P.wallT;
    if (abs(abs(p.x) - P.wallX) < lim) {
      let side = sign(p.x);
      p.x = select(side * (P.wallX + lim), side * (P.wallX - lim), abs(p.x) < P.wallX);
    }
    if (abs(abs(p.z) - P.wallZ) < lim) {
      let side = sign(p.z);
      p.z = select(side * (P.wallZ + lim), side * (P.wallZ - lim), abs(p.z) < P.wallZ);
    }
  }

  if (!iSolid) {
    if (p.y < -3.0 || dot(p.xz, p.xz) > P.cullR * P.cullR) {
      flags[i] = 0u; // cull
    }
  }
  pos[i] = vec4f(p, pos[i].w);
}

// ============ velocity update ============
@compute @workgroup_size(256)
fn velocityUpdate(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= TOTAL || (flags[i] & F_ACTIVE) == 0u) { return; }
  var v = (pos[i].xyz - prev[i].xyz) / P.h;
  let damp = select(P.dampFluid, P.dampSolid, i < SOLID_N);
  v *= max(0.0, 1.0 - damp * P.h);
  let sp2 = dot(v, v);
  if (sp2 > P.maxSpeed * P.maxSpeed) { v *= P.maxSpeed / sqrt(sp2); }
  vel[i] = vec4f(v, 0.0);
}

// ============ XSPH viscosity (fluid) ============
@compute @workgroup_size(256)
fn xsph(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + SOLID_N;
  if (i >= TOTAL || (flags[i] & F_ACTIVE) == 0u) { return; }
  let pi = pos[i].xyz;
  let H = P.fluidH;
  let H2 = H * H;
  let c0 = vec3i(floor(pi / P.cellSize));
  var sum = vec3f(0.0);
  var m = 0.0;
  for (var dx = -1; dx <= 1; dx++) {
  for (var dy = -1; dy <= 1; dy++) {
  for (var dz = -1; dz <= 1; dz++) {
    let r = cellRange(cellHash(c0 + vec3i(dx, dy, dz)));
    for (var e = r.x; e < r.y; e++) {
      let j = atomicLoad(&gridA[TABLE + TABLE + e]);
      if (j == i || j < SOLID_N) { continue; }
      let dd = pos[j].xyz - pi;
      if (dot(dd, dd) >= H2) { continue; }
      sum += vel[j].xyz;
      m += 1.0;
    }
  }}}
  var v = vel[i].xyz;
  if (m > 0.0) { v += P.viscosity * (sum / m - v); }
  delta[i] = vec4f(v, m);
}

@compute @workgroup_size(256)
fn xsphApply(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + SOLID_N;
  if (i >= TOTAL || (flags[i] & F_ACTIVE) == 0u) { return; }
  vel[i] = vec4f(delta[i].xyz, 0.0);
}

// ============ emit (hose) ============
@compute @workgroup_size(64)
fn emit(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x;
  if (k >= P.emitCount) { return; }
  let slot = SOLID_N + ((P.emitStart + k) % MAX_FLUID);
  let phase = P.emitDir.w + f32(k) * 2.39996;
  let spacing = P.fluidRadius * 2.2;
  let rad = spacing * (0.4 + 1.6 * fract(f32(k) * 0.618034));
  let off = P.emitSide.xyz * (cos(phase) * rad) + P.emitUp.xyz * (sin(phase) * rad)
          + P.emitDir.xyz * (f32(k % 8u) * spacing * 0.5);
  let p = P.emitPos.xyz + off;
  // cheap per-particle jitter
  let j = vec3f(fract(sin(phase * 12.9898) * 43758.547) - 0.5,
                fract(sin(phase * 78.233) * 12543.123) - 0.5,
                fract(sin(phase * 39.425) * 26781.897) - 0.5) * 0.5;
  let v = P.emitDir.xyz * P.emitPos.w + j;
  pos[slot] = vec4f(p, pos[slot].w);
  prev[slot] = vec4f(p, 0.0);
  vel[slot] = vec4f(v, 0.0);
  flags[slot] = F_ACTIVE;
}

// ============ knife: cut constraints crossing the swept wedge ============
@compute @workgroup_size(256)
fn cut(@builtin(global_invocation_id) g: vec3u) {
  let ci = g.x;
  if (ci >= NUM_CONS) { return; }
  let c = cons[ci];
  if (c.w == 0u) { return; }
  let o = P.cutO.xyz;
  let n = cross(P.cutD1.xyz, P.cutD2.xyz);
  if (dot(n, n) < 1e-16) { return; }
  let a = pos[c.x].xyz - o;
  let b = pos[c.y].xyz - o;
  let sa = dot(a, n);
  let sb = dot(b, n);
  if (sa * sb >= 0.0) { return; }
  let t = sa / (sa - sb);
  let p = a + (b - a) * t;
  if (dot(cross(P.cutD1.xyz, p), n) < 0.0) { return; }
  if (dot(cross(p, P.cutD2.xyz), n) < 0.0) { return; }
  cons[ci].w = 0u;
}

// ============ pick: nearest solid along ray (atomicMin packed) ============
@compute @workgroup_size(256)
fn pick(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= SOLID_N || (flags[i] & F_ACTIVE) == 0u) { return; }
  let ro = P.pickO.xyz;
  let rd = P.pickD.xyz;
  let q = pos[i].xyz - ro;
  let t = dot(q, rd);
  if (t < 0.0 || t > 60.0) { return; }
  let perp = q - rd * t;
  if (dot(perp, perp) > P.pickO.w * P.pickO.w) { return; }
  // pack: distance (mm, 16 bits) << 16 | particle index (16 bits)
  let packed = (min(u32(t * 1000.0), 0xffffu) << 16u) | (i & 0xffffu);
  atomicMin(&gridA[S_PICK], packed);
}

// ============ grab select / release ============
@compute @workgroup_size(256)
fn grabSelect(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= SOLID_N || (flags[i] & F_ACTIVE) == 0u) { return; }
  let dd = pos[i].xyz - P.grabC.xyz;
  if (dot(dd, dd) < P.grabC.w * P.grabC.w) {
    flags[i] |= F_GRABBED;
  }
}

@compute @workgroup_size(256)
fn grabRelease(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i < SOLID_N) { flags[i] &= ~F_GRABBED; }
}

// ============ active fluid count (per frame, for UI) ============
@compute @workgroup_size(256)
fn countActive(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + SOLID_N;
  if (i >= TOTAL || (flags[i] & F_ACTIVE) == 0u) { return; }
  atomicAdd(&gridA[S_COUNT], 1u);
}
`;
