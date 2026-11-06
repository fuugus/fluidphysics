// All WGSL kernels for the unified particle sim — disphenoid-voxel edition.
// Gather-only formulation: every kernel writes only to its own slot (or a
// private delta), so no float atomics are needed anywhere.
//
// Index space of the particle buffers (pos/prev/vel/flags/delta/density):
//   [0, VERT_N)          lattice vertices — carry mass; springs + volume only
//   [BALL0, FLUID0)      voxel balls — one per disphenoid, at its centroid.
//                        Positions are DERIVED from the 4 corner vertices;
//                        they are the only solid things that collide.
//   [FLUID0, TOTAL)      water particles
// The spatial hash holds balls + water only. A contact push on a ball is
// handed to its 4 vertices (ballsToVerts), so collisions deform the lattice.
//
// Voxel records (VREC u32 each, in the gridS tail at TS_VOX):
//   0-3  vertex ids a,b,c,d — long lines are (a,b) and (c,d), the rest short
//   4    rest volume (f32; <= 0 means the voxel is dead: cut/torn)
//   5    colour, rgba8 unorm
//   6    body id
//   7    —
//   8-11 inverse rest orientation (quat xyzw), static
//   12-15 current orientation (quat xyzw), written per frame by voxelInfo
//   16-18 angular velocity (rad/s), written per frame by voxelInfo
//   19   —
// The voxel's linear position/velocity live in pos/vel at BALL0 + t.

export const VREC = 20;

export const makeShaderSource = (C) => /* wgsl */ `

// ---- constants baked at pipeline creation ----
const VERT_N: u32 = ${C.VERT_N}u;
const VOX_N: u32 = ${C.VOX_N}u;
const BALL0: u32 = ${C.VERT_N}u;
const FLUID0: u32 = ${C.VERT_N + C.VOX_N}u;
const MAX_FLUID: u32 = ${C.MAX_FLUID}u;
const TOTAL: u32 = ${C.TOTAL}u;
const TABLE: u32 = ${C.TABLE}u;          // hash table size (pow2)
const NUM_CONS: u32 = ${C.NUM_CONS}u;
const SCAN_BLOCKS: u32 = ${C.SCAN_BLOCKS}u; // TABLE / 256

// scratch slots at the tail of the gridA buffer (after counts|cursor|entries)
const SCRATCH: u32 = TABLE + TABLE + TOTAL;
const S_PICK: u32 = SCRATCH;       // packed atomicMin pick result
const S_COUNT: u32 = SCRATCH + 1u; // active fluid counter
// ball-ball contact pair cache (collected once per substep, iterated)
const C_COUNT: u32 = SCRATCH + 8u;          // per-ball neighbor count
const C_LIST: u32 = C_COUNT + VOX_N;        // per-ball neighbor ids
const C_K: u32 = 40u;                        // max cached neighbors

// voxel records + per-vertex voxel adjacency (CSR) in the gridS tail
const VREC: u32 = ${VREC}u;
const TS_VOX: u32 = ${C.TS_VOX}u;
const T_ADJ_OFF: u32 = TS_VOX + VOX_N * VREC;    // CSR offsets [VERT_N+1]
const T_ADJ_DATA: u32 = T_ADJ_OFF + VERT_N + 1u; // incident voxel ids

const F_ACTIVE: u32 = 1u;
const F_GRABBED: u32 = 2u;
const SQRT3: f32 = 1.7320508;

struct Params {
  h: f32,               // substep dt
  gravity: f32,
  solidRadius: f32,     // voxel ball radius
  fluidRadius: f32,

  cellSize: f32,
  fluidH: f32,
  restDensity: f32,
  stiffness: f32,

  nearStiffness: f32,
  viscosity: f32,
  compliance: f32,      // short-line compliance (long lines: / longInfluence)
  omega: f32,           // jacobi relaxation for lines

  plasticThreshold: f32,
  plasticRate: f32,
  tearGap: f32,         // line rips when the gap exceeds this many ball diameters
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
  sleepSpeed: f32,
  spacing: f32,         // lattice unit a: short line = a*sqrt(3), long = 2a

  emitCount: u32,
  emitStart: u32,
  solidViscosity: f32,  // line-velocity smoothing (damps internal jiggle)
  grabBody: u32,        // restrict grab to one body (0xffff = any)

  emitPos: vec4f,       // xyz + speed in w
  emitDir: vec4f,       // xyz + phase in w
  emitSide: vec4f,
  emitUp: vec4f,

  grabDelta: vec4f,     // per-substep move of the grab target
  cutO: vec4f,          // knife wedge: camera origin
  cutD1: vec4f,
  cutD2: vec4f,

  pickO: vec4f,         // pick ray origin, w = max perp dist
  pickD: vec4f,         // pick ray dir
  grabC: vec4f,         // grab-select sphere center, w = radius
  misc: vec4f,          // x: shellSeal, y: longInfluence, z: contactShare, w: frame dt
}

@group(0) @binding(0) var<storage, read_write> pos: array<vec4f>;     // xyz + invMass
@group(0) @binding(1) var<storage, read_write> prev: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> flags: array<u32>;     // bit0 active, bit1 grabbed, bits 8-15 body
@group(0) @binding(4) var<storage, read_write> delta: array<vec4f>;   // xyz + count
@group(0) @binding(5) var<storage, read_write> density: array<vec2f>;
@group(0) @binding(6) var<storage, read_write> cons: array<vec4u>;    // a, b, rest(f32 bits), type (0 dead, 1 short, 2 long)
@group(0) @binding(7) var<storage, read_write> adj: array<u32>;       // [VERT_N+1 offsets][line ids]
@group(0) @binding(8) var<storage, read_write> gridA: array<atomic<u32>>; // counts | cursor | entries | scratch
@group(0) @binding(9) var<storage, read_write> gridS: array<u32>;     // scan | voxel records | voxel CSR
@group(0) @binding(10) var<uniform> P: Params;

fn isBall(i: u32) -> bool { return i >= BALL0 && i < FLUID0; }
fn bodyOf(i: u32) -> u32 { return (flags[i] >> 8u) & 0xffu; }

fn vrec(t: u32) -> u32 { return TS_VOX + t * VREC; }
fn voxAlive(t: u32) -> bool { return bitcast<f32>(gridS[vrec(t) + 4u]) > 0.0; }

fn cellHash(c: vec3i) -> u32 {
  let h = (u32(c.x) * 92837111u) ^ (u32(c.y) * 689287499u) ^ (u32(c.z) * 283923481u);
  return h & (TABLE - 1u);
}

fn posHash(p: vec3f) -> u32 {
  return cellHash(vec3i(floor(p / P.cellSize)));
}

// ground + basin walls as a position clamp (no friction)
fn envClamp(p0: vec3f, r: f32) -> vec3f {
  var p = p0;
  if (p.y < r) { p.y = r; }
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
  return p;
}

// ============ integrate (vertices + water; balls are derived) ============
@compute @workgroup_size(256)
fn integrate(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  // vertices only: balls are derived, water is integrated by MPM (mpm.js)
  if (i >= VERT_N || (flags[i] & F_ACTIVE) == 0u) { return; }
  var v = vel[i].xyz;
  v.y += P.gravity * P.h;
  let p = pos[i].xyz;
  var np = p + v * P.h;
  var tw = prev[i].w;
  // a vertex whose voxels were all cut away has nothing to collide with —
  // retire it once it has fallen out of the world
  if (i < VERT_N && np.y < -3.0) { flags[i] = 0u; return; }
  if ((flags[i] & F_GRABBED) != 0u) {
    // spring toward the grab target, which follows the mouse. The target is
    // stashed in otherwise-unused w/x slots: (prev.w, vel.w, density.x)
    var t = vec3f(prev[i].w, vel[i].w, density[i].x);
    t += P.grabDelta.xyz;
    // grip slips when overstretched (piece ripped away from the held cluster)
    if (distance(np, t) > 1.0) {
      flags[i] &= ~F_GRABBED;
      prev[i] = vec4f(p, tw);
      pos[i] = vec4f(np, pos[i].w);
      vel[i] = vec4f(v, 0.0);
      return;
    }
    var corr = (t - np) * (P.grabK * density[i].y); // density.y = grip weight
    let cl = length(corr);
    let maxC = 0.05;
    if (cl > maxC) { corr *= maxC / cl; }
    np += corr;
    tw = t.x;
    vel[i] = vec4f(v, t.y);
    density[i] = vec2f(t.z, density[i].y);
  } else {
    vel[i] = vec4f(v, vel[i].w);
  }
  prev[i] = vec4f(p, tw);
  pos[i] = vec4f(np, pos[i].w);
}

// ============ voxel balls: centroid of the 4 corners ============
fn voxCenter(t: u32) {
  let i = BALL0 + t;
  let b = vrec(t);
  if (bitcast<f32>(gridS[b + 4u]) <= 0.0) { flags[i] &= ~F_ACTIVE; return; }
  let c = (pos[gridS[b]].xyz + pos[gridS[b + 1u]].xyz
         + pos[gridS[b + 2u]].xyz + pos[gridS[b + 3u]].xyz) * 0.25;
  pos[i] = vec4f(c, pos[i].w);
}

@compute @workgroup_size(256)
fn voxelCenters(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= VOX_N) { return; }
  voxCenter(g.x);
}

// once per substep after integrate: also the pre-step centroid (friction)
@compute @workgroup_size(256)
fn voxelCentersPrev(@builtin(global_invocation_id) g: vec3u) {
  let t = g.x;
  if (t >= VOX_N) { return; }
  voxCenter(t);
  let b = vrec(t);
  let c = (prev[gridS[b]].xyz + prev[gridS[b + 1u]].xyz
         + prev[gridS[b + 2u]].xyz + prev[gridS[b + 3u]].xyz) * 0.25;
  prev[BALL0 + t] = vec4f(c, prev[BALL0 + t].w);
}

// ============ cube <-> MPM water ============
fn cornerVel(t: u32) -> vec3f {
  let b = vrec(t);
  return (vel[gridS[b]].xyz + vel[gridS[b + 1u]].xyz + vel[gridS[b + 2u]].xyz + vel[gridS[b + 3u]].xyz) * 0.25;
}
// before the MPM steps: balls at their corners' predicted spot + velocity
@compute @workgroup_size(256)
fn voxelVel(@builtin(global_invocation_id) g: vec3u) {
  let t = g.x;
  if (t >= VOX_N) { return; }
  voxCenter(t);
  if (!voxAlive(t)) { return; }
  vel[BALL0 + t] = vec4f(cornerVel(t), 0.0);
}
// after: what the water changed in a ball's velocity becomes a displacement
@compute @workgroup_size(256)
fn ballWaterPush(@builtin(global_invocation_id) g: vec3u) {
  let t = g.x;
  if (t >= VOX_N) { return; }
  var d = vec3f(0.0);
  if (voxAlive(t)) { d = (vel[BALL0 + t].xyz - cornerVel(t)) * P.h; }
  delta[BALL0 + t] = vec4f(d, 0.0);
}
// each corner takes the average push of its living voxels
@compute @workgroup_size(256)
fn ballAvgToVerts(@builtin(global_invocation_id) g: vec3u) {
  let v = g.x;
  if (v >= VERT_N || (flags[v] & F_ACTIVE) == 0u) { return; }
  var sum = vec3f(0.0);
  var n = 0.0;
  let start = gridS[T_ADJ_OFF + v];
  let end = gridS[T_ADJ_OFF + v + 1u];
  for (var k = start; k < end; k++) {
    let t = gridS[T_ADJ_DATA + k];
    if (!voxAlive(t)) { continue; }
    sum += delta[BALL0 + t].xyz;
    n += 1.0;
  }
  delta[v] = vec4f(sum / max(n, 1.0), 1.0);
}

// ============ grid build (balls + water) ============
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
  let i = g.x + BALL0;
  if (i >= TOTAL || (flags[i] & F_ACTIVE) == 0u) { return; }
  atomicAdd(&gridA[posHash(pos[i].xyz)], 1u);
}

// hierarchical exclusive scan of counts -> gridS[0..TABLE]
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
  if (i == 0u) { gridS[TABLE] = 0u; }
}

@compute @workgroup_size(256)
fn gridScatter(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + BALL0;
  if (i >= TOTAL || (flags[i] & F_ACTIVE) == 0u) { return; }
  let h = posHash(pos[i].xyz);
  let slot = gridS[h] + atomicAdd(&gridA[TABLE + h], 1u);
  atomicStore(&gridA[TABLE + TABLE + slot], i);
}

fn cellRange(h: u32) -> vec2u {
  let s = gridS[h];
  return vec2u(s, s + atomicLoad(&gridA[h]));
}

// ============ fluid: double-density relaxation ============
@compute @workgroup_size(256)
fn fluidDensity(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + FLUID0;
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
      if (j == i || j < FLUID0) { continue; }
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
  let i = g.x + FLUID0;
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
      if (j == i || j < FLUID0) { continue; }
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
  let i = g.x + FLUID0;
  if (i >= TOTAL || (flags[i] & F_ACTIVE) == 0u) { return; }
  pos[i] = vec4f(pos[i].xyz + delta[i].xyz, pos[i].w);
}

// ============ lines: XPBD jacobi over vertex adjacency ============
// Two line types with their own rest length (stored per line) and their own
// influence: long lines are softer by longInfluence (default sqrt(3)/2, the
// 1/length stiffness of a real rod), applied as compliance so the ratio holds
// regardless of iteration count.
@compute @workgroup_size(256)
fn solidSolve(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= VERT_N || (flags[i] & F_ACTIVE) == 0u) { return; }
  let pi = pos[i].xyz;
  let wi = pos[i].w;
  let alpha = P.compliance / (P.h * P.h);
  var corr = vec3f(0.0);
  let start = adj[i];
  let end = adj[i + 1u];
  // each PAIR is scaled by 1/max(deg_i, deg_j) — both endpoints use the same
  // factor, so Newton's third law holds exactly (own-count averaging injects
  // momentum at irregular surfaces → fragments "dance" forever)
  let degI = end - start;
  for (var k = start; k < end; k++) {
    let ci = adj[VERT_N + 1u + k];
    let c = cons[ci];
    if (c.w == 0u) { continue; }
    let other = select(c.x, c.y, c.x == i);
    let rest = bitcast<f32>(c.z);
    let dd = pi - pos[other].xyz;
    let d = length(dd);
    if (d < 1e-9) { continue; }
    let wo = pos[other].w;
    let wsum = wi + wo;
    if (wsum == 0.0) { continue; }
    let degJ = adj[other + 1u] - adj[other];
    let s = 1.0 / f32(max(degI, degJ));
    let inf = select(1.0, max(P.misc.y, 1e-4), c.w == 2u);
    let dl = -(d - rest) / (wsum + alpha / inf);
    corr += dd * (dl * wi / d * s);
  }
  delta[i] = vec4f(corr, 1.0);
}

@compute @workgroup_size(256)
fn applyDeltaSolid(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= VERT_N || (flags[i] & F_ACTIVE) == 0u) { return; }
  pos[i] = vec4f(pos[i].xyz + delta[i].xyz * P.omega, pos[i].w);
}

// clamped unit-scale apply for vertices (volume + contact deltas)
@compute @workgroup_size(256)
fn applyVertDelta(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= VERT_N || (flags[i] & F_ACTIVE) == 0u) { return; }
  var d = delta[i].xyz;
  let maxC = 0.5 * P.solidRadius; // per iteration; the loop runs many
  let m2 = dot(d, d);
  if (m2 > maxC * maxC) { d *= maxC / sqrt(m2); }
  pos[i] = vec4f(pos[i].xyz + d, pos[i].w);
}

// ============ line update: tearing + plasticity + voxel ownership ============
// Runs once per substep in line space, after the solve iterations.
fn lineRest0(kind: u32) -> f32 {
  return P.spacing * select(SQRT3, 2.0, kind == 2u);
}

@compute @workgroup_size(256)
fn bondUpdate(@builtin(global_invocation_id) g: vec3u) {
  let ci = g.x;
  if (ci >= NUM_CONS) { return; }
  let c = cons[ci];
  if (c.w == 0u) { return; }
  let d = distance(pos[c.x].xyz, pos[c.y].xyz);
  let rest = bitcast<f32>(c.z);
  // ripping: line breaks when the gap opens past tearGap ball diameters
  if (d > rest + P.tearGap * 2.0 * P.solidRadius) { cons[ci].w = 0u; return; }
  // a line exists only as an edge of a living voxel: once every voxel that
  // contains it is gone (cut / torn), it stops holding the pieces together
  var held = false;
  let s0 = gridS[T_ADJ_OFF + c.x];
  let s1 = gridS[T_ADJ_OFF + c.x + 1u];
  for (var k = s0; k < s1; k++) {
    let t = gridS[T_ADJ_DATA + k];
    if (!voxAlive(t)) { continue; }
    let b = vrec(t);
    if (gridS[b] == c.y || gridS[b + 1u] == c.y || gridS[b + 2u] == c.y || gridS[b + 3u] == c.y) {
      held = true;
      break;
    }
  }
  if (!held) { cons[ci].w = 0u; return; }
  // plasticity, clamped tightly around the pristine rest of the line's type
  let strain = (d - rest) / rest;
  var nr = rest;
  if (abs(strain) > P.plasticThreshold) { nr = rest + (d - rest) * P.plasticRate; }
  let r0 = lineRest0(c.w);
  nr = clamp(nr, 0.9 * r0, 1.25 * r0);
  if (nr != rest) { cons[ci].z = bitcast<u32>(nr); }
}

// ============ voxel volume constraints ============
// Each voxel's signed volume is held at its rest value (XPBD, zero
// compliance) — press a region and it bulges instead of compressing away.
@compute @workgroup_size(256)
fn solveTets(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= VERT_N || (flags[i] & F_ACTIVE) == 0u) { return; }
  var corr = vec3f(0.0);
  let start = gridS[T_ADJ_OFF + i];
  let end = gridS[T_ADJ_OFF + i + 1u];
  for (var k = start; k < end; k++) {
    let t = gridS[T_ADJ_DATA + k];
    let base = vrec(t);
    let ia = gridS[base];
    let ib = gridS[base + 1u];
    let ic = gridS[base + 2u];
    let id = gridS[base + 3u];
    let V0 = bitcast<f32>(gridS[base + 4u]);
    if (V0 <= 0.0) { continue; } // dead voxel
    let pa = pos[ia].xyz; let pb = pos[ib].xyz;
    let pc = pos[ic].xyz; let pd = pos[id].xyz;
    let V = dot(cross(pb - pa, pc - pa), pd - pa) / 6.0;
    let C = V - V0;
    let gb = cross(pc - pa, pd - pa) / 6.0;
    let gc = cross(pd - pa, pb - pa) / 6.0;
    let gd = cross(pb - pa, pc - pa) / 6.0;
    let ga = -(gb + gc + gd);
    let w = pos[ia].w; // vertices share inv mass
    let denom = w * (dot(ga, ga) + dot(gb, gb) + dot(gc, gc) + dot(gd, gd));
    if (denom < 1e-12) { continue; }
    let dl = -C / denom;
    // identical pair scale for all four endpoints: 1/max(incident voxels)
    let dm = max(max(gridS[T_ADJ_OFF + ia + 1u] - gridS[T_ADJ_OFF + ia],
                     gridS[T_ADJ_OFF + ib + 1u] - gridS[T_ADJ_OFF + ib]),
                 max(gridS[T_ADJ_OFF + ic + 1u] - gridS[T_ADJ_OFF + ic],
                     gridS[T_ADJ_OFF + id + 1u] - gridS[T_ADJ_OFF + id]));
    let s = 1.5 / f32(max(dm, 1u)); // over-relaxed — the voxels are well-conditioned
    var gMine = ga;
    if (i == ib) { gMine = gb; } else if (i == ic) { gMine = gc; } else if (i == id) { gMine = gd; }
    var c1 = gMine * (w * dl * s);
    let cl2 = dot(c1, c1);
    let lim = 0.5 * P.solidRadius; // per-voxel contribution cap (degenerate voxels)
    if (cl2 > lim * lim) { c1 *= lim / sqrt(cl2); }
    corr += c1;
  }
  delta[i] = vec4f(corr, 1.0);
}

// kill voxels whose edges overstretch (same rule as line ripping)
@compute @workgroup_size(256)
fn tetUpdate(@builtin(global_invocation_id) g: vec3u) {
  let t = g.x;
  if (t >= VOX_N) { return; }
  let base = vrec(t);
  let V0 = bitcast<f32>(gridS[base + 4u]);
  if (V0 <= 0.0) { return; }
  let gap = P.tearGap * 2.0 * P.solidRadius;
  for (var p = 0u; p < 3u; p++) {
    for (var q = p + 1u; q < 4u; q++) {
      let a = gridS[base + p];
      let b = gridS[base + q];
      let d = distance(pos[a].xyz, pos[b].xyz);
      let isLong = (p == 0u && q == 1u) || (p == 2u && q == 3u);
      if (d > lineRest0(select(1u, 2u, isLong)) + gap) {
        gridS[base + 4u] = bitcast<u32>(-V0);
        return;
      }
    }
  }
}

// knife: kill voxels whose edges cross the swept wedge
@compute @workgroup_size(256)
fn cutTets(@builtin(global_invocation_id) g: vec3u) {
  let t = g.x;
  if (t >= VOX_N) { return; }
  let base = vrec(t);
  let V0 = bitcast<f32>(gridS[base + 4u]);
  if (V0 <= 0.0) { return; }
  let o = P.cutO.xyz;
  let n = cross(P.cutD1.xyz, P.cutD2.xyz);
  if (dot(n, n) < 1e-16) { return; }
  for (var p = 0u; p < 3u; p++) {
    for (var q = p + 1u; q < 4u; q++) {
      let pa = pos[gridS[base + p]].xyz - o;
      let pb = pos[gridS[base + q]].xyz - o;
      let sa = dot(pa, n);
      let sb = dot(pb, n);
      if (sa * sb >= 0.0) { continue; }
      let tt = sa / (sa - sb);
      let pp = pa + (pb - pa) * tt;
      if (dot(cross(P.cutD1.xyz, pp), n) < 0.0) { continue; }
      if (dot(cross(pp, P.cutD2.xyz), n) < 0.0) { continue; }
      gridS[base + 4u] = bitcast<u32>(-V0);
      return;
    }
  }
}

// ============ ball-ball contacts: cache + iterated resolution ============
// Voxels that share a vertex are lattice neighbours and never collide (their
// balls overlap by design). Voxels that share no vertex sit >= 2a apart at
// rest, so with ball radius < a an intact body has zero internal contacts —
// and anything that does touch is a genuinely separate piece or a fold.
fn sharesVertex(t: u32, u: u32) -> bool {
  let a = vrec(t);
  let b = vrec(u);
  for (var p = 0u; p < 4u; p++) {
    let v = gridS[a + p];
    if (v == gridS[b] || v == gridS[b + 1u] || v == gridS[b + 2u] || v == gridS[b + 3u]) { return true; }
  }
  return false;
}

@compute @workgroup_size(256)
fn collectContacts(@builtin(global_invocation_id) g: vec3u) {
  let t = g.x;
  if (t >= VOX_N) { return; }
  let i = BALL0 + t;
  var cnt = 0u;
  if ((flags[i] & F_ACTIVE) != 0u) {
    let pi = pos[i].xyz;
    let rsum = 2.0 * P.solidRadius;
    // margin covers per-substep approach; the 27-cell gather reaches cellSize
    let lim = max(rsum, min(rsum * 2.0, P.cellSize));
    let c0 = vec3i(floor(pi / P.cellSize));
    for (var dx = -1; dx <= 1; dx++) {
    for (var dy = -1; dy <= 1; dy++) {
    for (var dz = -1; dz <= 1; dz++) {
      let r = cellRange(cellHash(c0 + vec3i(dx, dy, dz)));
      for (var e = r.x; e < r.y; e++) {
        let j = atomicLoad(&gridA[TABLE + TABLE + e]);
        if (j == i || j >= FLUID0 || cnt >= C_K) { continue; }
        let dd = pi - pos[j].xyz;
        if (dot(dd, dd) >= lim * lim) { continue; }
        if (sharesVertex(t, j - BALL0)) { continue; }
        atomicStore(&gridA[C_LIST + t * C_K + cnt], j);
        cnt++;
      }
    }}}
  }
  atomicStore(&gridA[C_COUNT + t], cnt);
}

@compute @workgroup_size(256)
fn ballContacts(@builtin(global_invocation_id) g: vec3u) {
  let t = g.x;
  if (t >= VOX_N) { return; }
  let i = BALL0 + t;
  if ((flags[i] & F_ACTIVE) == 0u) { return; }
  let pi = pos[i].xyz;
  let wi = pos[i].w;
  let rsum = 2.0 * P.solidRadius;
  var corr = vec3f(0.0);
  let cnt = atomicLoad(&gridA[C_COUNT + t]);
  for (var k = 0u; k < cnt; k++) {
    let j = atomicLoad(&gridA[C_LIST + t * C_K + k]);
    if ((flags[j] & F_ACTIVE) == 0u) { continue; }
    let dd = pi - pos[j].xyz;
    let d2 = dot(dd, dd);
    if (d2 >= rsum * rsum || d2 < 1e-12) { continue; }
    let wj = pos[j].w;
    let d = sqrt(d2);
    corr += dd * ((rsum - d) / d * wi / (wi + wj) * 0.5);
  }
  // ground + walls (static): full push
  corr += envClamp(pi, P.solidRadius) - pi;
  delta[i] = vec4f(corr, 0.0);
}

// hand every living ball's push to its 4 corners. A fixed share per voxel
// (contactShare) keeps momentum exact: each ball's coupling mass is defined
// as 4 * share * vertexMass, so what the ball gives, its corners receive.
@compute @workgroup_size(256)
fn ballsToVerts(@builtin(global_invocation_id) g: vec3u) {
  let v = g.x;
  if (v >= VERT_N || (flags[v] & F_ACTIVE) == 0u) { return; }
  var sum = vec3f(0.0);
  let start = gridS[T_ADJ_OFF + v];
  let end = gridS[T_ADJ_OFF + v + 1u];
  for (var k = start; k < end; k++) {
    let t = gridS[T_ADJ_DATA + k];
    if (!voxAlive(t)) { continue; }
    sum += delta[BALL0 + t].xyz;
  }
  delta[v] = vec4f(sum * P.misc.z, 1.0);
}

// once per substep: ground friction + wall push for balls
@compute @workgroup_size(256)
fn ballBounds(@builtin(global_invocation_id) g: vec3u) {
  let t = g.x;
  if (t >= VOX_N) { return; }
  let i = BALL0 + t;
  if ((flags[i] & F_ACTIVE) == 0u) { delta[i] = vec4f(0.0); return; }
  let r = P.solidRadius;
  let p = pos[i].xyz;
  var corr = envClamp(p, r) - p;
  if (p.y < r * 1.05) {
    let pv = prev[i].xyz;
    corr.x -= (p.x - pv.x) * P.frictionSolid;
    corr.z -= (p.z - pv.z) * P.frictionSolid;
  }
  delta[i] = vec4f(corr, 0.0);
}

// ============ contacts: ball-water coupling ============
// Two passes. A water particle typically overlaps several (overlapping)
// voxel balls at once; resolving each pair in full would throw it back
// several times its penetration — a super-elastic bounce whose recoil kicks
// the body. So each pair is scaled by 1/max(count_i, count_j): both sides use
// the same factor (momentum exact) and nobody overshoots.
fn contactRsum(iSolid: bool, jSolid: bool) -> f32 {
  return select(P.fluidRadius, P.solidRadius, iSolid) + select(P.fluidRadius, P.solidRadius, jSolid) + P.misc.x;
}

@compute @workgroup_size(256)
fn contactCount(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + BALL0;
  if (i >= TOTAL || (flags[i] & F_ACTIVE) == 0u) { return; }
  let iSolid = i < FLUID0;
  let pi = pos[i].xyz;
  let c0 = vec3i(floor(pi / P.cellSize));
  var cnt = 0.0;
  for (var dx = -1; dx <= 1; dx++) {
  for (var dy = -1; dy <= 1; dy++) {
  for (var dz = -1; dz <= 1; dz++) {
    let r = cellRange(cellHash(c0 + vec3i(dx, dy, dz)));
    for (var e = r.x; e < r.y; e++) {
      let j = atomicLoad(&gridA[TABLE + TABLE + e]);
      if (j == i || (j < FLUID0) == iSolid) { continue; }
      let rsum = contactRsum(iSolid, !iSolid);
      let dd = pi - pos[j].xyz;
      if (dot(dd, dd) < rsum * rsum) { cnt += 1.0; }
    }
  }}}
  density[i] = vec2f(cnt, density[i].y); // free slot for balls + MPM water
}

@compute @workgroup_size(256)
fn contacts(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + BALL0;
  if (i >= TOTAL || (flags[i] & F_ACTIVE) == 0u) { return; }
  let iSolid = i < FLUID0;
  let pi = pos[i].xyz;
  let wi = pos[i].w;
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
      let jSolid = j < FLUID0;
      // water-water is DDR's job; ball-ball lives in the iterated cache
      if (iSolid == jSolid) { continue; }
      let dd = pi - pos[j].xyz;
      let rj = select(P.fluidRadius, P.solidRadius, jSolid);
      let rsum = ri + rj + P.misc.x;
      let d2 = dot(dd, dd);
      if (d2 >= rsum * rsum || d2 < 1e-12) { continue; }
      let wj = pos[j].w;
      let wsum = wi + wj;
      if (wsum == 0.0) { continue; }
      let d = sqrt(d2);
      let s = 1.0 / max(1.0, max(density[i].x, density[j].x));
      corr += dd * ((rsum - d) / d * wi / wsum * s);
      cnt += 1.0;
    }
  }}}
  delta[i] = vec4f(corr, cnt);
}

// vertices (deltas from ballsToVerts) + water; balls are derived, skipped
@compute @workgroup_size(256)
fn applyDeltaAll(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= TOTAL || isBall(i) || (flags[i] & F_ACTIVE) == 0u) { return; }
  var d = delta[i].xyz;
  // safety cap only: must exceed any single contact's correction, or the
  // water side gets cut short while the body side doesn't (momentum leak)
  let r = select(P.fluidRadius + P.solidRadius + P.misc.x, P.solidRadius, i < VERT_N);
  let m2 = dot(d, d);
  if (m2 > r * r) { d *= r / sqrt(m2); }
  pos[i] = vec4f(pos[i].xyz + d, pos[i].w);
}

// ============ bounds (water; balls handle the solids) ============
@compute @workgroup_size(256)
fn bounds(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + FLUID0;
  if (i >= TOTAL || (flags[i] & F_ACTIVE) == 0u) { return; }
  var p = pos[i].xyz;
  let pv = prev[i].xyz;
  let r = P.fluidRadius;
  if (p.y < r) {
    p.x -= (p.x - pv.x) * P.frictionFluid;
    p.z -= (p.z - pv.z) * P.frictionFluid;
  }
  p = envClamp(p, r);
  if (p.y < -3.0 || dot(p.xz, p.xz) > P.cullR * P.cullR) {
    flags[i] = 0u; // cull
  }
  pos[i] = vec4f(p, pos[i].w);
}

// water: remember the post-MPM position, so velocityUpdate can tell what the
// ball contacts + bounds did to it this substep
@compute @workgroup_size(256)
fn fluidMark(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + FLUID0;
  if (i >= TOTAL || (flags[i] & F_ACTIVE) == 0u) { return; }
  prev[i] = vec4f(pos[i].xyz, prev[i].w);
}

// ============ velocity update (vertices + water) ============
@compute @workgroup_size(256)
fn velocityUpdate(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= TOTAL || isBall(i) || (flags[i] & F_ACTIVE) == 0u) { return; }
  if (i >= FLUID0) {
    // MPM owns the water velocity; add the push from ball contacts on top
    var v = vel[i].xyz + (pos[i].xyz - prev[i].xyz) / P.h;
    let s2 = dot(v, v);
    if (s2 > P.maxSpeed * P.maxSpeed) { v *= P.maxSpeed / sqrt(s2); }
    vel[i] = vec4f(v, vel[i].w); // w = J
    return;
  }
  var v = (pos[i].xyz - prev[i].xyz) / P.h;
  let damp = select(P.dampFluid, P.dampSolid, i < VERT_N);
  v *= max(0.0, 1.0 - damp * P.h);
  let sp2 = dot(v, v);
  if (sp2 > P.maxSpeed * P.maxSpeed) { v *= P.maxSpeed / sqrt(sp2); }
  vel[i] = vec4f(v, vel[i].w); // w carries the grab target's y component
}

// ============ line-velocity smoothing (vertices) ============
@compute @workgroup_size(256)
fn solidVisc(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= VERT_N || (flags[i] & F_ACTIVE) == 0u) { return; }
  var sum = vec3f(0.0);
  var m = 0.0;
  let start = adj[i];
  let end = adj[i + 1u];
  for (var k = start; k < end; k++) {
    let c = cons[adj[VERT_N + 1u + k]];
    if (c.w == 0u) { continue; }
    let other = select(c.x, c.y, c.x == i);
    sum += vel[other].xyz;
    m += 1.0;
  }
  var v = vel[i].xyz;
  if (m > 0.0) { v += P.solidViscosity * (sum / m - v); }
  delta[i] = vec4f(v, 0.0);
}

@compute @workgroup_size(256)
fn solidViscApply(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= VERT_N || (flags[i] & F_ACTIVE) == 0u) { return; }
  vel[i] = vec4f(delta[i].xyz, vel[i].w);
}

// ============ XSPH viscosity (fluid) ============
@compute @workgroup_size(256)
fn xsph(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + FLUID0;
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
      if (j == i || j < FLUID0) { continue; }
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
  let i = g.x + FLUID0;
  if (i >= TOTAL || (flags[i] & F_ACTIVE) == 0u) { return; }
  vel[i] = vec4f(delta[i].xyz, 0.0);
}

// ============ emit (hose) ============
@compute @workgroup_size(64)
fn emit(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x;
  if (k >= P.emitCount) { return; }
  let slot = FLUID0 + ((P.emitStart + k) % MAX_FLUID);
  let phase = P.emitDir.w + f32(k) * 2.39996;
  let spacing = P.fluidRadius * 2.2;
  let rad = spacing * (0.4 + 1.6 * fract(f32(k) * 0.618034));
  let off = P.emitSide.xyz * (cos(phase) * rad) + P.emitUp.xyz * (sin(phase) * rad)
          + P.emitDir.xyz * (f32(k % 8u) * spacing * 0.5);
  let p = P.emitPos.xyz + off;
  let j = vec3f(fract(sin(phase * 12.9898) * 43758.547) - 0.5,
                fract(sin(phase * 78.233) * 12543.123) - 0.5,
                fract(sin(phase * 39.425) * 26781.897) - 0.5) * 0.5;
  let v = P.emitDir.xyz * P.emitPos.w + j;
  pos[slot] = vec4f(p, pos[slot].w);
  prev[slot] = vec4f(p, 0.0);
  vel[slot] = vec4f(v, -1.0); // w < 0: fresh MPM particle (J = 1, C = 0)
  flags[slot] = F_ACTIVE;
}

// ============ knife: cut lines crossing the swept wedge ============
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

// ============ pick: nearest voxel ball along ray (atomicMin packed) ============
@compute @workgroup_size(256)
fn pick(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + BALL0;
  if (i >= FLUID0 || (flags[i] & F_ACTIVE) == 0u) { return; }
  let ro = P.pickO.xyz;
  let rd = P.pickD.xyz;
  let q = pos[i].xyz - ro;
  let t = dot(q, rd);
  if (t < 0.0 || t > 60.0) { return; }
  let perp = q - rd * t;
  if (dot(perp, perp) > P.pickO.w * P.pickO.w) { return; }
  // pack: distance (mm, 16 bits) << 16 | body id (16 bits)
  let packed = (min(u32(t * 1000.0), 0xffffu) << 16u) | bodyOf(i);
  atomicMin(&gridA[S_PICK], packed);
}

// ============ grab select / release (vertices carry the mass) ============
@compute @workgroup_size(256)
fn grabSelect(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= VERT_N || (flags[i] & F_ACTIVE) == 0u) { return; }
  // a grab holds exactly one body — never both sides of a contact
  if (P.grabBody != 0xffffu && bodyOf(i) != P.grabBody) { return; }
  let dd = pos[i].xyz - P.grabC.xyz;
  let q2 = dot(dd, dd) / (P.grabC.w * P.grabC.w);
  if (q2 < 1.0) {
    flags[i] |= F_GRABBED;
    // firm core, elastic skirt
    let w = (1.0 - q2) * (1.0 - q2) * 0.95 + 0.05;
    let p = pos[i].xyz;
    prev[i] = vec4f(prev[i].xyz, p.x);
    vel[i] = vec4f(vel[i].xyz, p.y);
    density[i] = vec2f(p.z, w);
  }
}

@compute @workgroup_size(256)
fn grabRelease(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i < VERT_N) { flags[i] &= ~F_GRABBED; }
}

// ============ active fluid count (per frame, for UI) ============
@compute @workgroup_size(256)
fn countActive(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + FLUID0;
  if (i >= TOTAL || (flags[i] & F_ACTIVE) == 0u) { return; }
  atomicAdd(&gridA[S_COUNT], 1u);
}

// ============ voxel info (per frame): velocity, orientation, spin ============
// A disphenoid's two long lines are perpendicular, so they give each voxel a
// clean orthonormal frame — orientation without any polar decomposition.
fn quatFromBasis(a: vec3f, b: vec3f, c: vec3f) -> vec4f {
  // columns a, b, c of a rotation matrix → quaternion (x, y, z, w)
  let m00 = a.x; let m10 = a.y; let m20 = a.z;
  let m01 = b.x; let m11 = b.y; let m21 = b.z;
  let m02 = c.x; let m12 = c.y; let m22 = c.z;
  let tr = m00 + m11 + m22;
  var q: vec4f;
  if (tr > 0.0) {
    let s = sqrt(tr + 1.0) * 2.0;
    q = vec4f((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s);
  } else if (m00 > m11 && m00 > m22) {
    let s = sqrt(1.0 + m00 - m11 - m22) * 2.0;
    q = vec4f(0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s);
  } else if (m11 > m22) {
    let s = sqrt(1.0 + m11 - m00 - m22) * 2.0;
    q = vec4f((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s);
  } else {
    let s = sqrt(1.0 + m22 - m00 - m11) * 2.0;
    q = vec4f((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s);
  }
  return normalize(q);
}

fn qmul(a: vec4f, b: vec4f) -> vec4f {
  return vec4f(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

struct Frame { e1: vec3f, e2: vec3f, e3: vec3f }
fn voxFrame(x0: vec3f, x1: vec3f, x2: vec3f, x3: vec3f) -> Frame {
  let e1 = normalize(x1 - x0 + vec3f(1e-9, 0.0, 0.0));
  let l2 = x3 - x2;
  let e2 = normalize(l2 - e1 * dot(e1, l2) + vec3f(0.0, 1e-9, 0.0));
  return Frame(e1, e2, cross(e1, e2));
}

@compute @workgroup_size(256)
fn voxelInfo(@builtin(global_invocation_id) g: vec3u) {
  let t = g.x;
  if (t >= VOX_N) { return; }
  let b = vrec(t);
  if (bitcast<f32>(gridS[b + 4u]) <= 0.0) { return; }
  let i = BALL0 + t;
  let ia = gridS[b]; let ib = gridS[b + 1u]; let ic = gridS[b + 2u]; let id = gridS[b + 3u];
  let x0 = pos[ia].xyz; let x1 = pos[ib].xyz; let x2 = pos[ic].xyz; let x3 = pos[id].xyz;
  let v0 = vel[ia].xyz; let v1 = vel[ib].xyz; let v2 = vel[ic].xyz; let v3 = vel[id].xyz;
  vel[i] = vec4f((v0 + v1 + v2 + v3) * 0.25, 0.0);

  let F = voxFrame(x0, x1, x2, x3);
  let q0inv = vec4f(bitcast<f32>(gridS[b + 8u]), bitcast<f32>(gridS[b + 9u]),
                    bitcast<f32>(gridS[b + 10u]), bitcast<f32>(gridS[b + 11u]));
  let q = qmul(quatFromBasis(F.e1, F.e2, F.e3), q0inv);
  gridS[b + 12u] = bitcast<u32>(q.x); gridS[b + 13u] = bitcast<u32>(q.y);
  gridS[b + 14u] = bitcast<u32>(q.z); gridS[b + 15u] = bitcast<u32>(q.w);

  // spin: frame a short dt ago (from corner velocities), w = ½ Σ e_prev × e
  let dt = max(P.misc.w, 1e-4);
  let Fp = voxFrame(x0 - v0 * dt, x1 - v1 * dt, x2 - v2 * dt, x3 - v3 * dt);
  let w = 0.5 * (cross(Fp.e1, F.e1) + cross(Fp.e2, F.e2) + cross(Fp.e3, F.e3)) / dt;
  gridS[b + 16u] = bitcast<u32>(w.x); gridS[b + 17u] = bitcast<u32>(w.y);
  gridS[b + 18u] = bitcast<u32>(w.z);
}
`;
