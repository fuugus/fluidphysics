import { makeShaderSource, VREC } from './wgsl.js';
import { MpmWater } from './mpm.js';

const b64encode = (buf) => {
  const u8 = new Uint8Array(buf);
  let s = '';
  for (let i = 0; i < u8.length; i += 0x8000)
    s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  return btoa(s);
};
const b64decode = (s) => {
  const bin = atob(s);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return u8.buffer;
};

// GPU port of the unified particle solver, disphenoid-voxel edition.
// Bodies are built from identical disphenoid voxels (the tetrahedral BCC
// lattice): vertices carry mass and are tied by short + long lines plus a
// volume constraint per voxel; each voxel is one colliding ball at its
// centroid. See wgsl.js for the buffer layout.

export const DEFAULTS = {
  SOLID_DIM: 11,                // body = n x n x n lattice cells (cell = 2a)
  SOLID_DIM2: null,             // small companion body (null = ~1/20 volume)
  MAX_FLUID: 65536,
  TABLE: 1 << 18,

  spacing: 0.05,                // lattice unit a: short line a*sqrt(3), long 2a
  solidMass: 0.01,              // per vertex
  fluidMass: 0.0015,
  cubeCenter: [0, 1.8, 0],

  substeps: 5,
  solidIters: 8,

  params: {
    h: 1 / 300,
    gravity: -10,
    solidRadius: 0.03,  // voxel ball radius
    fluidRadius: 0.05,
    cellSize: 0.13,
    fluidH: 0.13,
    restDensity: 3.0,
    stiffness: 0.0015,
    nearStiffness: 0.006,
    viscosity: 0.3,
    compliance: 5e-4,   // short lines; long lines get compliance / longInfluence
    longInfluence: Math.sqrt(3) / 2, // long-line stiffness relative to short (1/length rod)
    contactShare: 0.25, // fraction of a ball's contact push each corner vertex takes
    omega: 1.5,
    plasticThreshold: 0.38,
    plasticRate: 0.04,
    tearGap: 1.3,
    maxSpeed: 14,
    wallX: 3.5,
    wallZ: 3.5,
    wallH: 0.92,
    wallT: 0.07,
    cullR: 7.0,
    dampSolid: 1.2,
    dampFluid: 0.6,
    frictionSolid: 0.4,
    frictionFluid: 0.04,
    grabK: 0.6,
    sleepSpeed: 0,
    solidViscosity: 0,
  },
};

const KERNELS = [
  'integrate', 'voxelCenters', 'voxelCentersPrev', 'voxelVel', 'ballWaterPush', 'ballAvgToVerts',
  'gridClear', 'gridCount', 'scanBlocks', 'scanPartials', 'scanApply', 'gridScatter',
  'fluidDensity', 'fluidDisp', 'applyDeltaFluid',
  'solidSolve', 'applyDeltaSolid', 'applyVertDelta', 'bondUpdate',
  'solveTets', 'tetUpdate', 'cutTets',
  'collectContacts', 'ballContacts', 'ballsToVerts', 'ballBounds',
  'contactCount', 'contacts', 'applyDeltaAll', 'bounds', 'fluidMark', 'velocityUpdate',
  'solidVisc', 'solidViscApply',
  'xsph', 'xsphApply', 'emit', 'cut', 'pick', 'grabSelect', 'grabRelease', 'countActive',
  'voxelInfo',
];

// ---- small host-side math (mirrors the WGSL voxel frame) ----
const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm3 = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
function quatFromBasis(a, b, c) {
  const m00 = a[0], m10 = a[1], m20 = a[2];
  const m01 = b[0], m11 = b[1], m21 = b[2];
  const m02 = c[0], m12 = c[1], m22 = c[2];
  const tr = m00 + m11 + m22;
  let q;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    q = [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s];
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    q = [0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s];
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    q = [(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s];
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    q = [(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s];
  }
  const l = Math.hypot(...q);
  return q.map((v) => v / l);
}

// one body: every disphenoid of the BCC lattice whose centroid lies in the
// box [0, 2n]^3 (lattice units). Lattice points have all-even or all-odd
// integer coords; each tet is enumerated once via its odd-odd long line.
function disphenoidBox(n) {
  const add = (p, a, s) => { const q = p.slice(); q[a] += s; return q; };
  const tets = [];
  for (let x = -1; x <= 2 * n + 1; x += 2)
    for (let y = -1; y <= 2 * n + 1; y += 2)
      for (let z = -1; z <= 2 * n + 1; z += 2)
        for (let a = 0; a < 3; a++) {
          const b = (a + 1) % 3, d = (a + 2) % 3;
          const c1 = [x, y, z], c2 = add(c1, a, 2), m = add(c1, a, 1);
          for (const s of [-1, 1]) {
            for (const t of [
              [c1, c2, add(add(m, b, s), d, -1), add(add(m, b, s), d, 1)],
              [c1, c2, add(add(m, b, -1), d, s), add(add(m, b, 1), d, s)],
            ]) {
              const cen = [0, 1, 2].map((k) => (t[0][k] + t[1][k] + t[2][k] + t[3][k]) / 4);
              if (cen.every((v) => v >= 0 && v <= 2 * n)) tets.push(t);
            }
          }
        }
  return tets;
}

export class GpuSim {
  constructor(device, opts = {}) {
    this.device = device;
    const o = { ...DEFAULTS, ...opts };
    o.params = { ...DEFAULTS.params, ...(opts.params || {}) };
    this.o = o;
    this.DIM2 = o.SOLID_DIM2 ?? Math.max(2, Math.round(o.SOLID_DIM / Math.cbrt(20)));
    o.SOLID_DIM2 = this.DIM2; // resolved value lands in snapshots
    this.substeps = o.substeps;
    this.solidIters = o.solidIters;
    this.fluidNext = 0;
    this.frame = 0;
    this.pending = { emitCount: 0, cuts: [], grabDelta: [0, 0, 0], grabbing: false };
    this._buildHostData();
    this._buildGpu();
    this.mpm = new MpmWater(device, this, o.mpm);
  }

  // ---- initial voxel bodies + lines on CPU (uploaded once) ----
  _buildHostData() {
    const { SOLID_DIM: n, spacing: a, cubeCenter, solidMass, fluidMass } = this.o;
    const share = this.o.params.contactShare;
    const verts = [];   // { p: world xyz, body }
    const tets = [];    // { v: [4 ids], body }
    this.bodies = [];

    const addBody = (cells, center, body) => {
      const vmap = new Map();
      const vid = (p) => {
        const k = p.join();
        let v = vmap.get(k);
        if (v === undefined) {
          v = verts.length;
          vmap.set(k, v);
          verts.push({ p: p.map((c, i) => center[i] + (c - cells) * a), body });
        }
        return v;
      };
      const t0 = tets.length, v0 = verts.length;
      for (const t of disphenoidBox(cells)) tets.push({ v: t.map(vid), body });
      this.bodies.push({ body, cells, center, size: 2 * cells * a,
        vox0: t0, voxN: tets.length - t0, vert0: v0, vertN: verts.length - v0 });
    };

    addBody(n, cubeCenter, 0);
    const n2 = this.DIM2;
    if (n2 > 0) addBody(n2, [1.1, n2 * a + 0.5, 0.7], 1);

    const V = verts.length, T = tets.length;
    this.VERT_N = V;
    this.VOX_N = T;
    this.BALL0 = V;
    this.FLUID0 = V + T;
    this.TOTAL = V + T + this.o.MAX_FLUID;
    // compat names used by the renderers: solids end at FLUID0, the first
    // body's balls end at N1
    this.SOLID_N = this.FLUID0;
    this.N1 = V + this.bodies[0].voxN;

    const pos = new Float32Array(this.TOTAL * 4);
    const flags = new Uint32Array(this.TOTAL);
    for (let v = 0; v < V; v++) {
      pos.set([...verts[v].p, 1 / solidMass], v * 4);
      flags[v] = 1 | (verts[v].body << 8);
    }
    const P = (v) => verts[v].p;
    const signedVol = (t) => dot3(cross3(sub3(P(t[1]), P(t[0])), sub3(P(t[2]), P(t[0]))), sub3(P(t[3]), P(t[0])));

    // voxel records (see wgsl.js header for the layout)
    const rec = new Uint32Array(T * VREC);
    const recF = new Float32Array(rec.buffer);
    const BODY_COL = [[0.88, 0.12, 0.2], [0.15, 0.78, 0.45]];
    const hash = (k) => { const s = Math.sin(k * 12.9898) * 43758.5453; return s - Math.floor(s); };
    const ballInvMass = 1 / (4 * share * solidMass); // momentum-exact with ballsToVerts
    const lines = new Map(); // key -> { a, b, kind }
    tets.forEach((t, k) => {
      const v = t.v;
      // positive orientation; swapping c<->d keeps the long-line pairing
      if (signedVol(v) < 0) { const tmp = v[2]; v[2] = v[3]; v[3] = tmp; }
      const o = k * VREC;
      rec.set(v, o);
      recF[o + 4] = signedVol(v) / 6;
      const col = BODY_COL[t.body] || [0.8, 0.8, 0.8];
      const j = 0.9 + 0.2 * hash(k + 1); // slight per-voxel shade so voxels read
      rec[o + 5] = col.reduce((acc, c, i) => acc | (Math.round(Math.min(1, c * j) * 255) << (8 * i)), 0) | (255 << 24);
      rec[o + 6] = t.body;
      // inverse rest orientation from the two (perpendicular) long lines
      const e1 = norm3(sub3(P(v[1]), P(v[0])));
      const l2 = sub3(P(v[3]), P(v[2]));
      const e2 = norm3(sub3(l2, e1.map((c) => c * dot3(e1, l2))));
      const q0 = quatFromBasis(e1, e2, cross3(e1, e2));
      recF.set([-q0[0], -q0[1], -q0[2], q0[3]], o + 8);
      recF.set([0, 0, 0, 1], o + 12);
      // ball: centroid, coupling mass, body id
      const ci = (V + k) * 4;
      for (let c = 0; c < 3; c++) pos[ci + c] = (P(v[0])[c] + P(v[1])[c] + P(v[2])[c] + P(v[3])[c]) / 4;
      pos[ci + 3] = ballInvMass;
      flags[V + k] = 1 | (t.body << 8);
      // the 6 lines: (a,b) and (c,d) are long, the other 4 short
      for (const [p, q] of [[0, 1], [2, 3], [0, 2], [0, 3], [1, 2], [1, 3]]) {
        const lo = Math.min(v[p], v[q]), hi = Math.max(v[p], v[q]);
        const key = lo * 4194304 + hi;
        if (!lines.has(key)) lines.set(key, { a: lo, b: hi, kind: (p === 0 && q === 1) || (p === 2 && q === 3) ? 2 : 1 });
      }
    });
    for (let i = this.FLUID0; i < this.TOTAL; i++) pos[i * 4 + 3] = 1 / fluidMass;
    this.initialFluid = 0;

    // lines + per-vertex CSR adjacency
    const lineArr = [...lines.values()];
    this.NUM_CONS = lineArr.length;
    const cons = new Uint32Array(this.NUM_CONS * 4);
    const consF = new Float32Array(cons.buffer);
    const incident = Array.from({ length: V }, () => []);
    lineArr.forEach((l, c) => {
      cons[c * 4] = l.a;
      cons[c * 4 + 1] = l.b;
      consF[c * 4 + 2] = Math.hypot(...sub3(P(l.a), P(l.b)));
      cons[c * 4 + 3] = l.kind;
      incident[l.a].push(c); incident[l.b].push(c);
    });
    const adj = new Uint32Array(V + 1 + 2 * this.NUM_CONS);
    let off = 0;
    for (let i = 0; i < V; i++) {
      adj[i] = off;
      for (const c of incident[i]) adj[V + 1 + off++] = c;
    }
    adj[V] = off;

    // per-vertex voxel adjacency (CSR)
    const tetInc = Array.from({ length: V }, () => []);
    tets.forEach((t, k) => { for (const v of t.v) tetInc[v].push(k); });
    const tetAdj = new Uint32Array(V + 1 + 4 * T);
    let toff = 0;
    for (let i = 0; i < V; i++) {
      tetAdj[i] = toff;
      for (const k of tetInc[i]) tetAdj[V + 1 + toff++] = k;
    }
    tetAdj[V] = toff;

    // face neighbours: faceNbr[t*4+f] = voxel across the face opposite
    // corner f, or -1 on the body's outer surface (tetravoxel renderer)
    const faceNbr = new Int32Array(T * 4).fill(-1);
    const faceOwner = new Map();
    tets.forEach((t, k) => {
      for (let f = 0; f < 4; f++) {
        const key = t.v.filter((_, j) => j !== f).sort((x, y) => x - y).join(',');
        const other = faceOwner.get(key);
        if (other === undefined) faceOwner.set(key, k * 4 + f);
        else { faceNbr[k * 4 + f] = other >> 2; faceNbr[other] = k; }
      }
    });

    this.init = { pos, flags, cons, adj, rec, tetAdj };
    // kept for the renderers (barycentric embedding, tetravoxel faces)
    this.host = {
      vertRest: Float32Array.from(verts.flatMap((v) => v.p)),
      tetVerts: Uint32Array.from(tets.flatMap((t) => t.v)),
      faceNbr,
    };
  }

  _buildGpu() {
    const d = this.device;
    const { TABLE, MAX_FLUID } = this.o;
    const N = this.TOTAL;
    const V = this.VERT_N, T = this.VOX_N;
    const mk = (size, extraUsage = 0) => d.createBuffer({
      size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extraUsage,
    });
    // voxel records start on a 256-byte boundary so renderers can bind them
    this.TS_VOX = Math.ceil((TABLE + 1 + TABLE / 256) / 64) * 64;
    this.buf = {
      pos: mk(N * 16),
      prev: mk(N * 16),
      vel: mk(N * 16),
      flags: mk(N * 4),
      delta: mk(N * 16),
      density: mk(N * 8),
      cons: mk(this.NUM_CONS * 16),
      adj: mk((V + 1 + 2 * this.NUM_CONS) * 4),
      gridA: mk((TABLE * 2 + N + 8 + T * 41) * 4), // + ball contact cache
      // gridS: [scan][pad][voxel records][voxel CSR offsets][voxel CSR data]
      gridS: mk((this.TS_VOX + T * VREC + V + 1 + T * 4) * 4),
      params: d.createBuffer({ size: 320, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    };
    this.voxRecOffset = this.TS_VOX * 4;     // bytes, for render bindings
    this.voxRecSize = T * VREC * 4;
    this.staging = {
      pick: d.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
      count: d.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
      read: d.createBuffer({ size: N * 36, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
    };

    const code = makeShaderSource({
      VERT_N: V, VOX_N: T, MAX_FLUID, TOTAL: N, TABLE,
      NUM_CONS: this.NUM_CONS, SCAN_BLOCKS: TABLE / 256, TS_VOX: this.TS_VOX,
    });
    const module = d.createShaderModule({ code });

    const entries = [];
    for (let b = 0; b < 10; b++)
      entries.push({ binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } });
    entries.push({ binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } });
    this.bgl = d.createBindGroupLayout({ entries });
    const layout = d.createPipelineLayout({ bindGroupLayouts: [this.bgl] });

    this.pipes = {};
    for (const k of KERNELS)
      this.pipes[k] = d.createComputePipeline({ layout, compute: { module, entryPoint: k } });

    const order = ['pos', 'prev', 'vel', 'flags', 'delta', 'density', 'cons', 'adj', 'gridA', 'gridS'];
    this.bg = d.createBindGroup({
      layout: this.bgl,
      entries: [
        ...order.map((name, b) => ({ binding: b, resource: { buffer: this.buf[name] } })),
        { binding: 10, resource: { buffer: this.buf.params } },
      ],
    });

    this.paramsBuf = new ArrayBuffer(320);
    this.pf = new Float32Array(this.paramsBuf);
    this.pu = new Uint32Array(this.paramsBuf);

    this.resetSolid();
    this.device.queue.writeBuffer(this.buf.flags, 0, this.init.flags);
    this.device.queue.writeBuffer(this.buf.adj, 0, this.init.adj);
    this.device.queue.writeBuffer(this.buf.gridS, (this.TS_VOX + T * VREC) * 4, this.init.tetAdj);
  }

  // solids only — water keeps flowing through a reset
  resetSolid() {
    const q = this.device.queue;
    const S = this.FLUID0;
    q.writeBuffer(this.buf.pos, 0, this.init.pos, 0, S * 4);
    q.writeBuffer(this.buf.prev, 0, this.init.pos, 0, S * 4);
    q.writeBuffer(this.buf.vel, 0, new Float32Array(S * 4));
    q.writeBuffer(this.buf.flags, 0, this.init.flags, 0, S);
    q.writeBuffer(this.buf.cons, 0, this.init.cons);
    q.writeBuffer(this.buf.density, 0, new Float32Array(S * 2));
    q.writeBuffer(this.buf.gridS, this.TS_VOX * 4, this.init.rec);
  }

  clearFluid() {
    this.device.queue.writeBuffer(this.buf.flags, this.FLUID0 * 4,
      new Uint32Array(this.o.MAX_FLUID));
  }

  // ---- per-frame uniform upload ----
  _writeParams() {
    const p = this.o.params, pf = this.pf, pu = this.pu, pend = this.pending;
    const scalars = [
      p.h, p.gravity, p.solidRadius, p.fluidRadius,
      p.cellSize, p.fluidH, p.restDensity, p.stiffness,
      p.nearStiffness, p.viscosity, p.compliance, p.omega,
      p.plasticThreshold, p.plasticRate, p.tearGap, p.maxSpeed,
      p.wallX, p.wallZ, p.wallH, p.wallT,
      p.cullR, p.dampSolid, p.dampFluid, p.frictionSolid,
      p.frictionFluid, p.grabK, p.sleepSpeed, this.o.spacing,
    ];
    pf.set(scalars, 0);
    pu[28] = pend.emitCount;
    pu[29] = pend.emitStart || 0;
    pf[30] = p.solidViscosity;
    pu[31] = pend.grabBody ?? 0xffff;
    const v4 = (o, arr) => pf.set(arr, o);
    v4(32, pend.emitPos || [0, 0, 0, 0]);
    v4(36, pend.emitDir || [0, 0, 0, 0]);
    v4(40, pend.emitSide || [0, 0, 0, 0]);
    v4(44, pend.emitUp || [0, 0, 0, 0]);
    const gd = pend.grabDelta;
    v4(48, [gd[0] / this.substeps, gd[1] / this.substeps, gd[2] / this.substeps, 0]);
    v4(52, pend.cutO || [0, 0, 0, 0]);
    v4(56, pend.cutD1 || [0, 0, 0, 0]);
    v4(60, pend.cutD2 || [0, 0, 0, 0]);
    v4(64, pend.pickO || [0, 0, 0, 0.25]);
    v4(68, pend.pickD || [0, 0, 1, 0]);
    v4(72, pend.grabC || [0, 0, 0, 0]);
    v4(76, [p.shellSeal || 0, p.longInfluence, p.contactShare, 1 / 60]);
    this.device.queue.writeBuffer(this.buf.params, 0, this.paramsBuf);
    if (this.mpm) this.mpm.writeParams();
  }

  // ---- encode one frame of simulation into an encoder ----
  encode(encoder, timestamps = null) {
    let cutThisFrame = false;
    if (this.pending.cuts.length > 0) {
      const c = this.pending.cuts.shift();
      this.pending.cutO = c.o; this.pending.cutD1 = c.d1; this.pending.cutD2 = c.d2;
      cutThisFrame = true;
    }
    this._writeParams();
    this.pending.grabDelta = [0, 0, 0]; // consumed this frame
    const { TABLE } = this.o;
    const V = this.VERT_N, T = this.VOX_N, HASHED = this.TOTAL - this.BALL0;
    const wg = (n) => Math.ceil(n / 256);
    const pass = encoder.beginComputePass(timestamps ? {
      timestampWrites: { querySet: timestamps.set, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
    } : {});
    pass.setBindGroup(0, this.bg);
    const run = (name, n) => { pass.setPipeline(this.pipes[name]); pass.dispatchWorkgroups(n); };
    const ballPush = () => { run('ballsToVerts', wg(V)); run('applyVertDelta', wg(V)); };

    if (this.pending.emitCount > 0) run('emit', Math.ceil(this.pending.emitCount / 64));

    for (let s = 0; s < this.substeps; s++) {
      run('integrate', wg(V));
      // water: MPM steps on its own grid, with the cubes' voxel balls in it
      // as a second material; the water's push comes back to the corners
      run('voxelVel', wg(T));
      this.mpm.encode(pass);
      pass.setBindGroup(0, this.bg);
      run('ballWaterPush', wg(T));
      run('ballAvgToVerts', wg(V));
      run('applyVertDelta', wg(V));
      run('fluidMark', wg(this.o.MAX_FLUID));
      run('voxelCentersPrev', wg(T));
      run('gridClear', wg(TABLE));
      run('gridCount', wg(HASHED));
      run('scanBlocks', TABLE / 256);
      run('scanPartials', 1);
      run('scanApply', TABLE / 256);
      run('gridScatter', wg(HASHED));
      // unified loop: lines, volume, and ball contacts negotiate every iteration
      run('collectContacts', wg(T));
      for (let it = 0; it < this.solidIters; it++) {
        run('solidSolve', wg(V));
        run('applyDeltaSolid', wg(V));
        run('solveTets', wg(V));
        run('applyVertDelta', wg(V));
        run('solveTets', wg(V));          // volume gets a double dose:
        run('applyVertDelta', wg(V));     // bulk must outrun any press
        run('voxelCenters', wg(T));
        run('ballContacts', wg(T));
        ballPush();
      }
      run('bondUpdate', wg(this.NUM_CONS));
      run('tetUpdate', wg(T));
      run('voxelCenters', wg(T));
      // (water <-> cube coupling happens in the MPM grid, see mpm.js; the old
      // particle-contact kernels contacts/contactCount are no longer run)
      run('voxelCenters', wg(T));
      run('ballBounds', wg(T));           // walls + ground friction
      ballPush();
      run('bounds', wg(this.o.MAX_FLUID));
      run('velocityUpdate', wg(this.TOTAL));
      if (this.o.params.solidViscosity > 0) {
        run('solidVisc', wg(V));
        run('solidViscApply', wg(V));
      }
    }
    if (cutThisFrame) {
      run('cut', wg(this.NUM_CONS));
      run('cutTets', wg(T));
    }
    run('voxelCenters', wg(T));           // final ball positions for rendering
    run('voxelInfo', wg(T));
    run('countActive', wg(this.o.MAX_FLUID));
    pass.end();

    if (this.pending.emitCount > 0) {
      this.fluidNext = (this.fluidNext + this.pending.emitCount) % this.o.MAX_FLUID;
      this.pending.emitCount = 0;
    }
  }

  emit(count, posArr, dirArr, sideArr, upArr, speed, phase) {
    this.pending.emitCount = count;
    this.pending.emitStart = this.fluidNext;
    this.pending.emitPos = [...posArr, speed];
    this.pending.emitDir = [...dirArr, phase];
    this.pending.emitSide = [...sideArr, 0];
    this.pending.emitUp = [...upArr, 0];
  }

  queueCut(o, d1, d2) {
    this.pending.cuts.push({ o: [...o, 0], d1: [...d1, 0], d2: [...d2, 0] });
  }

  // pick nearest voxel ball along ray; resolves to {body, t} or null
  async pick(ro, rd, maxPerp = 0.25) {
    if (this._pickBusy) return null;
    this._pickBusy = true;
    this.pending.pickO = [...ro, maxPerp];
    this.pending.pickD = [...rd, 0];
    this._writeParams();
    const SCRATCH = this.o.TABLE * 2 + this.TOTAL;
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setBindGroup(0, this.bg);
    pass.setPipeline(this.pipes.gridClear);
    pass.dispatchWorkgroups(Math.ceil(this.o.TABLE / 256));
    pass.setPipeline(this.pipes.pick);
    pass.dispatchWorkgroups(Math.ceil(this.VOX_N / 256));
    pass.end();
    enc.copyBufferToBuffer(this.buf.gridA, SCRATCH * 4, this.staging.pick, 0, 4);
    this.device.queue.submit([enc.finish()]);
    await this.staging.pick.mapAsync(GPUMapMode.READ);
    const packed = new Uint32Array(this.staging.pick.getMappedRange())[0];
    this.staging.pick.unmap();
    this._pickBusy = false;
    if (packed === 0xffffffff) return null;
    return { body: packed & 0xffff, t: (packed >>> 16) / 1000 };
  }

  grabAt(center, radius, body = 0xffff) {
    this.pending.grabC = [...center, radius];
    this.pending.grabBody = body;
    this.pending.grabbing = true;
    this._writeParams();
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setBindGroup(0, this.bg);
    pass.setPipeline(this.pipes.grabSelect);
    pass.dispatchWorkgroups(Math.ceil(this.VERT_N / 256));
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  release() {
    this.pending.grabbing = false;
    this.pending.grabDelta = [0, 0, 0];
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setBindGroup(0, this.bg);
    pass.setPipeline(this.pipes.grabRelease);
    pass.dispatchWorkgroups(Math.ceil(this.VERT_N / 256));
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  async readActiveCount() {
    if (this._countBusy) return this._lastCount ?? 0;
    this._countBusy = true;
    const SCRATCH = this.o.TABLE * 2 + this.TOTAL;
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.buf.gridA, (SCRATCH + 1) * 4, this.staging.count, 0, 4);
    this.device.queue.submit([enc.finish()]);
    await this.staging.count.mapAsync(GPUMapMode.READ);
    const n = new Uint32Array(this.staging.count.getMappedRange())[0];
    this.staging.count.unmap();
    this._countBusy = false;
    this._lastCount = n;
    return n;
  }

  async _readBufferAt(buf, offset, size) {
    const st = this.device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, offset, st, 0, size);
    this.device.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const data = st.getMappedRange().slice(0);
    st.destroy();
    return data;
  }
  _readBuffer(buf, size) { return this._readBufferAt(buf, 0, size); }

  // the voxel list: per voxel { verts, alive, color, body, pos, vel, quat, angVel }
  async readVoxels() {
    const T = this.VOX_N;
    const rec = await this._readBufferAt(this.buf.gridS, this.voxRecOffset, this.voxRecSize);
    const p = new Float32Array(await this._readBufferAt(this.buf.pos, this.BALL0 * 16, T * 16));
    const v = new Float32Array(await this._readBufferAt(this.buf.vel, this.BALL0 * 16, T * 16));
    const u = new Uint32Array(rec), f = new Float32Array(rec);
    const out = new Array(T);
    for (let t = 0; t < T; t++) {
      const o = t * VREC;
      const c = u[o + 5];
      out[t] = {
        verts: [u[o], u[o + 1], u[o + 2], u[o + 3]],
        alive: f[o + 4] > 0,
        color: [c & 255, (c >> 8) & 255, (c >> 16) & 255].map((x) => x / 255),
        body: u[o + 6],
        pos: [p[t * 4], p[t * 4 + 1], p[t * 4 + 2]],
        vel: [v[t * 4], v[t * 4 + 1], v[t * 4 + 2]],
        quat: [f[o + 12], f[o + 13], f[o + 14], f[o + 15]],
        angVel: [f[o + 16], f[o + 17], f[o + 18]],
      };
    }
    return out;
  }

  // full sim state → JSON-serializable object (Space key downloads this)
  async snapshot() {
    const N = this.TOTAL;
    const read = (name, size) => this._readBuffer(this.buf[name], size).then(b64encode);
    return {
      version: 2,
      app: 'slice-splash-voxels',
      opts: { ...this.o, params: { ...this.o.params } },
      fluidNext: this.fluidNext,
      numCons: this.NUM_CONS,
      buffers: {
        pos: await read('pos', N * 16),
        prev: await read('prev', N * 16),
        vel: await read('vel', N * 16),
        flags: await read('flags', N * 4),
        density: await read('density', N * 8),
        cons: await read('cons', this.NUM_CONS * 16),
        voxels: b64encode(await this._readBufferAt(this.buf.gridS, this.voxRecOffset, this.voxRecSize)),
      },
    };
  }

  restore(snap, keepGrabs = false) {
    const q = this.device.queue;
    if (snap.app !== 'slice-splash-voxels') throw new Error('snapshot is from the cubic-lattice version');
    for (const [name, data] of Object.entries(snap.buffers)) {
      const buf = b64decode(data);
      if (name === 'voxels') {
        if (buf.byteLength === this.voxRecSize) q.writeBuffer(this.buf.gridS, this.voxRecOffset, buf);
        continue;
      }
      if (name === 'flags' && !keepGrabs) {
        // standalone snapshot: clear grab bits (no release will ever come)
        const f = new Uint32Array(buf);
        for (let i = 0; i < f.length; i++) f[i] &= ~2;
      }
      q.writeBuffer(this.buf[name], 0, buf);
    }
    this.fluidNext = snap.fluidNext;
  }

  dispose() {
    this.mpm.dispose();
    for (const b of Object.values(this.buf)) b.destroy();
    for (const b of Object.values(this.staging)) b.destroy();
  }

  // count alive lines (test/debug)
  async readConsAlive() {
    const u = new Uint32Array(await this._readBuffer(this.buf.cons, this.NUM_CONS * 16));
    let alive = 0;
    for (let c = 0; c < this.NUM_CONS; c++) alive += u[c * 4 + 3] !== 0 ? 1 : 0;
    return { alive, total: this.NUM_CONS };
  }

  // full readback for tests/tuning (slow; not used in the render loop)
  async readState() {
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.buf.pos, 0, this.staging.read, 0, this.TOTAL * 16);
    enc.copyBufferToBuffer(this.buf.vel, 0, this.staging.read, this.TOTAL * 16, this.TOTAL * 16);
    enc.copyBufferToBuffer(this.buf.flags, 0, this.staging.read, this.TOTAL * 32, this.TOTAL * 4);
    this.device.queue.submit([enc.finish()]);
    await this.staging.read.mapAsync(GPUMapMode.READ);
    const raw = this.staging.read.getMappedRange().slice(0);
    this.staging.read.unmap();
    const data = new Float32Array(raw);
    return {
      pos: data.subarray(0, this.TOTAL * 4),
      vel: data.subarray(this.TOTAL * 4, this.TOTAL * 8),
      flags: new Uint32Array(raw, this.TOTAL * 32, this.TOTAL),
    };
  }
}
