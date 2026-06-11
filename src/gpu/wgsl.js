// All WGSL kernels for the unified particle sim.
// Gather-only formulation: every kernel writes only to its own particle's
// slot (or a private delta), so no float atomics are needed anywhere.

export const makeShaderSource = (C) => /* wgsl */ `

// ---- constants baked at pipeline creation ----
const SOLID_N: u32 = ${C.SOLID_N}u;
const N_DIM: u32 = ${C.SOLID_DIM}u;
const N1: u32 = ${C.N1}u;           // particle count of the first lattice
const N_DIM2: u32 = ${Math.max(1, C.SOLID_DIM2)}u;
const MAX_FLUID: u32 = ${C.MAX_FLUID}u;
const TOTAL: u32 = ${C.TOTAL}u;
const TABLE: u32 = ${C.TABLE}u;          // hash table size (pow2)
const NUM_CONS: u32 = ${C.NUM_CONS}u;
const SCAN_BLOCKS: u32 = ${C.SCAN_BLOCKS}u; // TABLE / 256

// scratch slots at the tail of the gridA buffer (after counts|cursor|entries)
const SCRATCH: u32 = TABLE + TABLE + TOTAL;
const S_PICK: u32 = SCRATCH;       // packed atomicMin pick result
const S_COUNT: u32 = SCRATCH + 1u; // active fluid counter
// solid-contact pair cache (collected once per substep, iterated with bonds)
const C_COUNT: u32 = SCRATCH + 8u;          // per-solid neighbor count
const C_LIST: u32 = C_COUNT + SOLID_N;      // per-solid neighbor ids
const C_K: u32 = 40u;                        // max cached neighbors

// tet volume data packed in the gridS tail
const NUM_TETS: u32 = ${C.NUM_TETS}u;
const TS_TETS: u32 = ${C.TS_TETS}u;                  // [a,b,c,d,restVol] x NUM_TETS
const T_ADJ_OFF: u32 = TS_TETS + NUM_TETS * 5u;      // CSR offsets [SOLID_N+1]
const T_ADJ_DATA: u32 = T_ADJ_OFF + SOLID_N + 1u;    // incident tet ids

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
  tearGap: f32,         // bond rips when the gap exceeds this many diameters
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
  sleepSpeed: f32,      // solids below this speed are put to rest
  spacing: f32,

  emitCount: u32,
  emitStart: u32,
  solidViscosity: f32,  // bonded-velocity smoothing (damps internal jiggle)
  grabLattice: u32,     // restrict grab to one lattice (0xffff = any)

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

// grabbed particles keep their normal mass — the grab is a spring (applied
// in integrate), not a kinematic pin, so held bodies dangle, swing, and can
// be thrown, and ripping only happens from genuine overstretch
fn effInvMass(i: u32) -> f32 {
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
  v.y += P.gravity * P.h;
  let p = pos[i].xyz;
  var np = p + v * P.h;
  var tw = prev[i].w;
  if ((flags[i] & F_GRABBED) != 0u) {
    // spring toward the grab target, which follows the mouse. The target is
    // stashed in otherwise-unused w/x slots: (prev.w, vel.w, density.x)
    var t = vec3f(prev[i].w, vel[i].w, density[i].x);
    t += P.grabDelta.xyz;
    // grip slips when overstretched — releases stragglers whose piece was
    // ripped away from the grabbed cluster (they'd otherwise keep towing
    // their body toward the cursor forever)
    if (distance(np, t) > 1.0) {
      flags[i] &= ~F_GRABBED;
      prev[i] = vec4f(p, tw);
      pos[i] = vec4f(np, pos[i].w);
      vel[i] = vec4f(v, 0.0);
      return;
    }
    var corr = (t - np) * (P.grabK * density[i].y); // density.y = grip weight
    let cl = length(corr);
    // caps follow speed — must stay below the per-substep contact resolution
    // capacity (iters x clamp) and inside the collect margin
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
  let start = adj[i];
  let end = adj[i + 1u];
  // each PAIR is scaled by 1/max(deg_i, deg_j) — both endpoints use the same
  // factor, so Newton's third law holds exactly. Dividing by each particle's
  // own count instead (classic Jacobi averaging) silently injects momentum
  // at every irregular surface, making torn fragments "dance" forever.
  let degI = end - start;
  for (var k = start; k < end; k++) {
    let ci = adj[SOLID_N + 1u + k];
    let c = cons[ci];
    if (c.w == 0u) { continue; }
    let other = select(c.x, c.y, c.x == i);
    let rest = bitcast<f32>(c.z);
    let dd = pi - pos[other].xyz;
    let d = length(dd);
    if (d < 1e-9) { continue; }
    let wo = effInvMass(other);
    let wsum = wi + wo;
    if (wsum == 0.0) { continue; }
    let degJ = adj[other + 1u] - adj[other];
    let s = 1.0 / f32(max(degI, degJ));
    let dl = -(d - rest) / (wsum + alpha);
    corr += dd * (dl * wi / d * s);
  }
  delta[i] = vec4f(corr, 1.0);
}

@compute @workgroup_size(256)
fn applyDeltaSolid(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= SOLID_N || (flags[i] & F_ACTIVE) == 0u) { return; }
  pos[i] = vec4f(pos[i].xyz + delta[i].xyz * P.omega, pos[i].w);
}

// ============ bond update: tearing + plasticity ============
// Runs once per substep in constraint space, AFTER the solve iterations.
// Doing this inside the solve gather raced: one endpoint could tear a bond
// while the other still applied its correction — a one-sided impulse that
// pumped momentum into ripped fragments.
fn latticeRest(a0: u32, b0: u32) -> f32 {
  // bonds never cross lattices, so both endpoints share the same indexing
  var a = a0; var b = b0; var D = N_DIM;
  if (a0 >= N1) { a = a0 - N1; b = b0 - N1; D = N_DIM2; }
  let ax = f32(a / (D * D)); let ay = f32((a / D) % D); let az = f32(a % D);
  let bx = f32(b / (D * D)); let by = f32((b / D) % D); let bz = f32(b % D);
  let dx = ax - bx; let dy = ay - by; let dz = az - bz;
  return P.spacing * sqrt(dx * dx + dy * dy + dz * dz);
}

@compute @workgroup_size(256)
fn bondUpdate(@builtin(global_invocation_id) g: vec3u) {
  let ci = g.x;
  if (ci >= NUM_CONS) { return; }
  let c = cons[ci];
  if (c.w == 0u) { return; }
  let d = distance(pos[c.x].xyz, pos[c.y].xyz);
  let rest = bitcast<f32>(c.z);
  // ripping: bond breaks when the gap opens past tearGap diameters
  if (d > rest + P.tearGap * 2.0 * P.solidRadius) { cons[ci].w = 0u; return; }
  // plasticity, clamped tightly around the pristine lattice rest. A wide
  // band lets neighboring bonds creep to geometrically incompatible rests —
  // the solver then can never converge and its per-substep residual churn
  // becomes perpetual kinetic energy ("dancing" fragments). The clamp is
  // enforced every substep so corrupted states heal themselves.
  let strain = (d - rest) / rest;
  var nr = rest;
  if (abs(strain) > P.plasticThreshold) { nr = rest + (d - rest) * P.plasticRate; }
  let r0 = latticeRest(c.x, c.y);
  nr = clamp(nr, 0.9 * r0, 1.25 * r0);
  if (nr != rest) { cons[ci].z = bitcast<u32>(nr); }
}

// same lattice AND neighboring grid cells — i.e. a potential cut face,
// where the reduced contact distance must apply so severed neighbors can
// rest at lattice spacing without fighting
fn latticeAdjacent(i: u32, j: u32) -> bool {
  if ((i >= N1) != (j >= N1)) { return false; }
  var a = i; var b = j; var D = N_DIM;
  if (i >= N1) { a = i - N1; b = j - N1; D = N_DIM2; }
  let ax = i32(a / (D * D)); let ay = i32((a / D) % D); let az = i32(a % D);
  let bx = i32(b / (D * D)); let by = i32((b / D) % D); let bz = i32(b % D);
  return abs(ax - bx) <= 1 && abs(ay - by) <= 1 && abs(az - bz) <= 1;
}

// ============ tet volume constraints ============
// Each lattice cell is 5 tets whose signed volume is held at its rest
// value (XPBD, zero compliance). This is what conserves volume: press a
// region and its tets can only comply by expanding perpendicular — the
// body bulges instead of compressing away.
@compute @workgroup_size(256)
fn solveTets(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= SOLID_N || (flags[i] & F_ACTIVE) == 0u) { return; }
  var corr = vec3f(0.0);
  let start = gridS[T_ADJ_OFF + i];
  let end = gridS[T_ADJ_OFF + i + 1u];
  for (var k = start; k < end; k++) {
    let t = gridS[T_ADJ_DATA + k];
    let base = TS_TETS + t * 5u;
    let ia = gridS[base];
    let ib = gridS[base + 1u];
    let ic = gridS[base + 2u];
    let id = gridS[base + 3u];
    let V0 = bitcast<f32>(gridS[base + 4u]);
    if (V0 <= 0.0) { continue; } // dead tet (torn/cut)
    let pa = pos[ia].xyz; let pb = pos[ib].xyz;
    let pc = pos[ic].xyz; let pd = pos[id].xyz;
    let V = dot(cross(pb - pa, pc - pa), pd - pa) / 6.0;
    let C = V - V0;
    // volume gradients (sum to zero -> momentum conserved exactly when all
    // four endpoints apply the same scale)
    let gb = cross(pc - pa, pd - pa) / 6.0;
    let gc = cross(pd - pa, pb - pa) / 6.0;
    let gd = cross(pb - pa, pc - pa) / 6.0;
    let ga = -(gb + gc + gd);
    let w = pos[ia].w; // solids share inv mass
    let denom = w * (dot(ga, ga) + dot(gb, gb) + dot(gc, gc) + dot(gd, gd));
    if (denom < 1e-12) { continue; }
    let dl = -C / denom;
    // identical pair scale for all four endpoints: 1/max(incident tets)
    let dm = max(max(gridS[T_ADJ_OFF + ia + 1u] - gridS[T_ADJ_OFF + ia],
                     gridS[T_ADJ_OFF + ib + 1u] - gridS[T_ADJ_OFF + ib]),
                 max(gridS[T_ADJ_OFF + ic + 1u] - gridS[T_ADJ_OFF + ic],
                     gridS[T_ADJ_OFF + id + 1u] - gridS[T_ADJ_OFF + id]));
    let s = 1.0 / f32(max(dm, 1u));
    var gMine = ga;
    if (i == ib) { gMine = gb; } else if (i == ic) { gMine = gc; } else if (i == id) { gMine = gd; }
    var c1 = gMine * (w * dl * s);
    let cl2 = dot(c1, c1);
    let lim = 0.5 * P.solidRadius; // per-tet contribution cap (degenerate tets)
    if (cl2 > lim * lim) { c1 *= lim / sqrt(cl2); }
    corr += c1;
  }
  delta[i] = vec4f(corr, 1.0);
}

// kill tets whose edges overstretch (same rule as bond ripping) — a cut or
// torn region must stop conserving volume across the gap
@compute @workgroup_size(256)
fn tetUpdate(@builtin(global_invocation_id) g: vec3u) {
  let t = g.x;
  if (t >= NUM_TETS) { return; }
  let base = TS_TETS + t * 5u;
  let V0 = bitcast<f32>(gridS[base + 4u]);
  if (V0 <= 0.0) { return; }
  let gap = P.tearGap * 2.0 * P.solidRadius;
  for (var p = 0u; p < 3u; p++) {
    for (var q = p + 1u; q < 4u; q++) {
      let a = gridS[base + p];
      let b = gridS[base + q];
      let d = distance(pos[a].xyz, pos[b].xyz);
      if (d > latticeRest(a, b) + gap) {
        gridS[base + 4u] = bitcast<u32>(-V0);
        return;
      }
    }
  }
}

// knife: kill tets whose edges cross the swept wedge
@compute @workgroup_size(256)
fn cutTets(@builtin(global_invocation_id) g: vec3u) {
  let t = g.x;
  if (t >= NUM_TETS) { return; }
  let base = TS_TETS + t * 5u;
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

// ============ solid contact cache + iterated resolution ============
// Collected once per substep with a margin, then resolved interleaved with
// the bond solve — bonds and contacts negotiate every iteration, so a
// driven body cannot out-iterate the collision response (it deforms or
// displaces the other body instead of tunneling into it).
fn solidRsum(i: u32, j: u32) -> f32 {
  let rsum = 2.0 * P.solidRadius;
  if (latticeAdjacent(i, j)) { return min(rsum, P.spacing * 0.97); }
  return rsum;
}

@compute @workgroup_size(256)
fn collectContacts(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= SOLID_N) { return; }
  var cnt = 0u;
  if ((flags[i] & F_ACTIVE) != 0u) {
    let pi = pos[i].xyz;
    // margin must exceed the fastest per-substep approach (grab drive +
    // solver drift), or pairs entering range mid-substep tunnel unseen
    let margin = 2.0 * P.solidRadius;
    let c0 = vec3i(floor(pi / P.cellSize));
    for (var dx = -1; dx <= 1; dx++) {
    for (var dy = -1; dy <= 1; dy++) {
    for (var dz = -1; dz <= 1; dz++) {
      let r = cellRange(cellHash(c0 + vec3i(dx, dy, dz)));
      for (var e = r.x; e < r.y; e++) {
        let j = atomicLoad(&gridA[TABLE + TABLE + e]);
        if (j == i || j >= SOLID_N || cnt >= C_K) { continue; }
        let dd = pi - pos[j].xyz;
        let lim = solidRsum(i, j) + margin;
        if (dot(dd, dd) >= lim * lim) { continue; }
        if (isBonded(i, j)) { continue; }
        atomicStore(&gridA[C_LIST + i * C_K + cnt], j);
        cnt++;
      }
    }}}
  }
  atomicStore(&gridA[C_COUNT + i], cnt);
}

@compute @workgroup_size(256)
fn solidContacts(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= SOLID_N || (flags[i] & F_ACTIVE) == 0u) { return; }
  let pi = pos[i].xyz;
  let wi = effInvMass(i);
  var corr = vec3f(0.0);
  let cnt = atomicLoad(&gridA[C_COUNT + i]);
  for (var k = 0u; k < cnt; k++) {
    let j = atomicLoad(&gridA[C_LIST + i * C_K + k]);
    let dd = pi - pos[j].xyz;
    let rsum = solidRsum(i, j);
    let d2 = dot(dd, dd);
    if (d2 >= rsum * rsum || d2 < 1e-12) { continue; }
    let wj = effInvMass(j);
    let wsum = wi + wj;
    if (wsum == 0.0) { continue; }
    let d = sqrt(d2);
    // hard relative to bonds: cross-body repulsion beats internal springs,
    // so the volume redistributes through the body instead of being entered
    corr += dd * ((rsum - d) / d * wi / wsum * 0.5);
  }
  delta[i] = vec4f(corr, 1.0);
}

@compute @workgroup_size(256)
fn applySolidContacts(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= SOLID_N || (flags[i] & F_ACTIVE) == 0u) { return; }
  var d = delta[i].xyz;
  let maxC = 0.5 * P.solidRadius; // per iteration; the loop runs many
  let m2 = dot(d, d);
  if (m2 > maxC * maxC) { d *= maxC / sqrt(m2); }
  pos[i] = vec4f(pos[i].xyz + d, pos[i].w);
}

// ============ contacts: solid-fluid coupling only ============
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
      // fluid-fluid is DDR's job; solid-solid lives in the iterated cache
      if (iSolid == jSolid) { continue; }
      let dd = pi - pos[j].xyz;
      let rj = select(P.fluidRadius, P.solidRadius, jSolid);
      let rsum = ri + rj;
      let d2 = dot(dd, dd);
      if (d2 >= rsum * rsum || d2 < 1e-12) { continue; }
      let wj = effInvMass(j);
      let wsum = wi + wj;
      if (wsum == 0.0) { continue; }
      let d = sqrt(d2);
      // fixed 0.5 pair scale (NOT averaged by own count) — both sides of a
      // contact must apply the same factor or momentum is not conserved
      corr += dd * ((rsum - d) / d * wi / wsum * 0.5);
      cnt += 1.0;
    }
  }}}
  delta[i] = vec4f(corr, cnt);
}

@compute @workgroup_size(256)
fn applyDeltaAll(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= TOTAL || (flags[i] & F_ACTIVE) == 0u) { return; }
  var d = delta[i].xyz;
  // safety clamp for pathological stacking (slightly breaks symmetry, but
  // only in extremes where stability matters more). Must stay ABOVE the
  // grab spring's per-substep drive, or sustained pushes tunnel through.
  let r = select(P.fluidRadius, P.solidRadius, i < SOLID_N);
  let m2 = dot(d, d);
  if (m2 > r * r) { d *= r / sqrt(m2); }
  pos[i] = vec4f(pos[i].xyz + d, pos[i].w);
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
  vel[i] = vec4f(v, vel[i].w); // w carries the grab target's y component
}

// ============ bonded-velocity smoothing (solids) ============
// XSPH over alive bonds: blends each particle's velocity toward its bonded
// neighborhood average. Damps internal oscillation modes (which the speed
// clamp otherwise sustains forever in ripped fragments) without touching
// positions — volume and rigid translation are preserved.
@compute @workgroup_size(256)
fn solidVisc(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= SOLID_N || (flags[i] & F_ACTIVE) == 0u) { return; }
  var sum = vec3f(0.0);
  var m = 0.0;
  let start = adj[i];
  let end = adj[i + 1u];
  for (var k = start; k < end; k++) {
    let c = cons[adj[SOLID_N + 1u + k]];
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
  if (i >= SOLID_N || (flags[i] & F_ACTIVE) == 0u) { return; }
  vel[i] = vec4f(delta[i].xyz, vel[i].w);
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
  // a grab holds exactly one body — never both sides of a contact, or the
  // spring drives them into each other and bypasses collision entirely
  let lat = select(0u, 1u, i >= N1);
  if (P.grabLattice != 0xffffu && lat != P.grabLattice) { return; }
  let dd = pos[i].xyz - P.grabC.xyz;
  let q2 = dot(dd, dd) / (P.grabC.w * P.grabC.w);
  if (q2 < 1.0) {
    flags[i] |= F_GRABBED;
    // grip strength falls off from the grab center to the sphere edge:
    // a firm core with an elastic skirt, so the grabbed region deforms
    // naturally instead of moving as a rigid plug
    let w = (1.0 - q2) * (1.0 - q2) * 0.95 + 0.05;
    // initialize the spring target at the particle's current position
    let p = pos[i].xyz;
    prev[i] = vec4f(prev[i].xyz, p.x);
    vel[i] = vec4f(vel[i].xyz, p.y);
    density[i] = vec2f(p.z, w);
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
