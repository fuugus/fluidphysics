import { makeShaderSource } from './wgsl.js';

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

// GPU port of the unified particle solver. All state lives in GPU buffers;
// the CPU only encodes passes and feeds small uniforms (tools, emitter).

export const DEFAULTS = {
  SOLID_DIM: 20,                // 20^3 = 8000 body particles
  MAX_FLUID: 65536,
  TABLE: 1 << 18,

  spacing: 0.065,
  solidMass: 0.004,
  fluidMass: 0.0015,
  cubeCenter: [0, 1.8, 0],

  substeps: 5,
  solidIters: 6,

  params: {
    h: 1 / 300,
    gravity: -10,
    solidRadius: 0.034,
    fluidRadius: 0.05,
    cellSize: 0.13,
    fluidH: 0.13,
    restDensity: 3.0,
    stiffness: 0.0015,
    nearStiffness: 0.006,
    viscosity: 0.3,
    compliance: 3e-5,
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
    sleepSpeed: 0,      // band-aids, retired since the momentum fix —
    solidViscosity: 0,  // params kept so they can be re-enabled for testing
  },
};

const KERNELS = [
  'integrate', 'gridClear', 'gridCount', 'scanBlocks', 'scanPartials', 'scanApply',
  'gridScatter', 'fluidDensity', 'fluidDisp', 'applyDeltaFluid', 'solidSolve', 'bondUpdate',
  'applyDeltaSolid', 'contacts', 'applyDeltaAll', 'bounds', 'velocityUpdate',
  'solidVisc', 'solidViscApply',
  'xsph', 'xsphApply', 'emit', 'cut', 'pick', 'grabSelect', 'grabRelease', 'countActive',
];

export class GpuSim {
  constructor(device, opts = {}) {
    this.device = device;
    const o = { ...DEFAULTS, ...opts };
    o.params = { ...DEFAULTS.params, ...(opts.params || {}) };
    this.o = o;
    this.SOLID_N = o.SOLID_DIM ** 3;
    this.TOTAL = this.SOLID_N + o.MAX_FLUID;
    this.substeps = o.substeps;
    this.solidIters = o.solidIters;
    this.fluidNext = 0;
    this.frame = 0;
    this.pending = { emitCount: 0, cuts: [], grabDelta: [0, 0, 0], grabbing: false };
    this._buildHostData();
    this._buildGpu();
  }

  // ---- initial lattice + constraints on CPU (uploaded once) ----
  _buildHostData() {
    const { SOLID_DIM: n, spacing, cubeCenter, solidMass, fluidMass, MAX_FLUID } = this.o;
    const N = this.SOLID_N;
    const pos = new Float32Array(this.TOTAL * 4);
    const flags = new Uint32Array(this.TOTAL);
    const half = (n - 1) * spacing / 2;
    let id = 0;
    for (let ix = 0; ix < n; ix++)
      for (let iy = 0; iy < n; iy++)
        for (let iz = 0; iz < n; iz++) {
          pos.set([
            cubeCenter[0] - half + ix * spacing,
            cubeCenter[1] - half + iy * spacing,
            cubeCenter[2] - half + iz * spacing,
            1 / solidMass,
          ], id * 4);
          flags[id] = 1;
          id++;
        }
    for (let i = N; i < this.TOTAL; i++) pos[i * 4 + 3] = 1 / fluidMass;

    // constraints: 26-neighborhood (within sqrt(3)*spacing)
    const idx = (ix, iy, iz) => (ix * n + iy) * n + iz;
    const consA = [], consB = [], consRest = [];
    const incident = Array.from({ length: N }, () => []);
    const maxD2 = 3 * spacing * spacing + 1e-9;
    for (let ix = 0; ix < n; ix++)
      for (let iy = 0; iy < n; iy++)
        for (let iz = 0; iz < n; iz++)
          for (let dx = -1; dx <= 1; dx++)
            for (let dy = -1; dy <= 1; dy++)
              for (let dz = -1; dz <= 1; dz++) {
                const jx = ix + dx, jy = iy + dy, jz = iz + dz;
                if (jx < 0 || jy < 0 || jz < 0 || jx >= n || jy >= n || jz >= n) continue;
                const a = idx(ix, iy, iz), b = idx(jx, jy, jz);
                if (b <= a) continue;
                const d2 = (dx * dx + dy * dy + dz * dz) * spacing * spacing;
                if (d2 > maxD2) continue;
                const c = consA.length;
                consA.push(a); consB.push(b); consRest.push(Math.sqrt(d2));
                incident[a].push(c); incident[b].push(c);
              }
    this.NUM_CONS = consA.length;

    const cons = new Uint32Array(this.NUM_CONS * 4);
    const consF = new Float32Array(cons.buffer);
    for (let c = 0; c < this.NUM_CONS; c++) {
      cons[c * 4] = consA[c];
      cons[c * 4 + 1] = consB[c];
      consF[c * 4 + 2] = consRest[c];
      cons[c * 4 + 3] = 1;
    }
    // CSR adjacency: [N+1 offsets][constraint ids]
    const adj = new Uint32Array(N + 1 + 2 * this.NUM_CONS);
    let off = 0;
    for (let i = 0; i < N; i++) {
      adj[i] = off;
      for (const c of incident[i]) adj[N + 1 + off++] = c;
    }
    adj[N] = off;

    this.init = { pos, flags, cons, adj };
  }

  _buildGpu() {
    const d = this.device;
    const { TABLE, MAX_FLUID } = this.o;
    const N = this.TOTAL;
    const mk = (size, extraUsage = 0) => d.createBuffer({
      size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extraUsage,
    });
    this.buf = {
      pos: mk(N * 16),
      prev: mk(N * 16),
      vel: mk(N * 16),
      flags: mk(N * 4),
      delta: mk(N * 16),
      density: mk(N * 8),
      cons: mk(this.NUM_CONS * 16),
      adj: mk((this.SOLID_N + 1 + 2 * this.NUM_CONS) * 4),
      gridA: mk((TABLE * 2 + N + 8) * 4),
      gridS: mk((TABLE + 1 + TABLE / 256) * 4),
      params: d.createBuffer({ size: 304, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    };
    this.staging = {
      pick: d.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
      count: d.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
      read: d.createBuffer({ size: N * 36, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
    };

    const code = makeShaderSource({
      SOLID_N: this.SOLID_N, SOLID_DIM: this.o.SOLID_DIM, MAX_FLUID, TOTAL: N, TABLE,
      NUM_CONS: this.NUM_CONS, SCAN_BLOCKS: TABLE / 256,
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

    // params CPU mirror
    this.paramsBuf = new ArrayBuffer(304);
    this.pf = new Float32Array(this.paramsBuf);
    this.pu = new Uint32Array(this.paramsBuf);

    this.resetSolid();
    this.device.queue.writeBuffer(this.buf.adj, 0, this.init.adj);
  }

  resetSolid() {
    const q = this.device.queue;
    q.writeBuffer(this.buf.pos, 0, this.init.pos);
    q.writeBuffer(this.buf.prev, 0, this.init.pos);
    q.writeBuffer(this.buf.vel, 0, new Float32Array(this.TOTAL * 4));
    q.writeBuffer(this.buf.flags, 0, this.init.flags.slice(0, this.SOLID_N), 0);
    q.writeBuffer(this.buf.cons, 0, this.init.cons);
    q.writeBuffer(this.buf.density, 0, new Float32Array(this.SOLID_N * 2));
  }

  clearFluid() {
    this.device.queue.writeBuffer(this.buf.flags, this.SOLID_N * 4,
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
    this.device.queue.writeBuffer(this.buf.params, 0, this.paramsBuf);
  }

  // ---- encode one frame of simulation into an encoder ----
  encode(encoder, timestamps = null) {
    // apply at most one queued knife stroke segment per frame
    let cutThisFrame = false;
    if (this.pending.cuts.length > 0) {
      const c = this.pending.cuts.shift();
      this.pending.cutO = c.o; this.pending.cutD1 = c.d1; this.pending.cutD2 = c.d2;
      cutThisFrame = true;
    }
    this._writeParams();
    this.pending.grabDelta = [0, 0, 0]; // consumed this frame
    const { TABLE } = this.o;
    const wg = (n) => Math.ceil(n / 256);
    const pass = encoder.beginComputePass(timestamps ? {
      timestampWrites: { querySet: timestamps.set, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
    } : {});
    pass.setBindGroup(0, this.bg);
    const run = (name, n) => { pass.setPipeline(this.pipes[name]); pass.dispatchWorkgroups(n); };

    if (this.pending.emitCount > 0) run('emit', Math.ceil(this.pending.emitCount / 64));

    for (let s = 0; s < this.substeps; s++) {
      run('integrate', wg(this.TOTAL));
      run('gridClear', wg(TABLE));
      run('gridCount', wg(this.TOTAL));
      run('scanBlocks', TABLE / 256);
      run('scanPartials', 1);
      run('scanApply', TABLE / 256);
      run('gridScatter', wg(this.TOTAL));
      run('fluidDensity', wg(this.o.MAX_FLUID));
      run('fluidDisp', wg(this.o.MAX_FLUID));
      run('applyDeltaFluid', wg(this.o.MAX_FLUID));
      for (let it = 0; it < this.solidIters; it++) {
        run('solidSolve', wg(this.SOLID_N));
        run('applyDeltaSolid', wg(this.SOLID_N));
      }
      run('bondUpdate', wg(this.NUM_CONS));
      run('contacts', wg(this.TOTAL));
      run('applyDeltaAll', wg(this.TOTAL));
      run('bounds', wg(this.TOTAL));
      run('velocityUpdate', wg(this.TOTAL));
      if (this.o.params.solidViscosity > 0) {
        run('solidVisc', wg(this.SOLID_N));
        run('solidViscApply', wg(this.SOLID_N));
      }
      run('xsph', wg(this.o.MAX_FLUID));
      run('xsphApply', wg(this.o.MAX_FLUID));
    }
    if (cutThisFrame) run('cut', wg(this.NUM_CONS));
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

  // pick nearest solid along ray; resolves to {id, t} or null
  async pick(ro, rd, maxPerp = 0.25) {
    if (this._pickBusy) return null;
    this._pickBusy = true;
    this.pending.pickO = [...ro, maxPerp];
    this.pending.pickD = [...rd, 0];
    this._writeParams();
    const SCRATCH = this.o.TABLE * 2 + this.TOTAL;
    const enc = this.device.createCommandEncoder();
    // S_PICK is reset by gridClear each substep; ensure it's fresh:
    const pass = enc.beginComputePass();
    pass.setBindGroup(0, this.bg);
    pass.setPipeline(this.pipes.gridClear);
    pass.dispatchWorkgroups(Math.ceil(this.o.TABLE / 256));
    pass.setPipeline(this.pipes.pick);
    pass.dispatchWorkgroups(Math.ceil(this.SOLID_N / 256));
    pass.end();
    enc.copyBufferToBuffer(this.buf.gridA, SCRATCH * 4, this.staging.pick, 0, 4);
    this.device.queue.submit([enc.finish()]);
    await this.staging.pick.mapAsync(GPUMapMode.READ);
    const packed = new Uint32Array(this.staging.pick.getMappedRange())[0];
    this.staging.pick.unmap();
    this._pickBusy = false;
    if (packed === 0xffffffff) return null;
    return { id: packed & 0xffff, t: (packed >>> 16) / 1000 };
  }

  grabAt(center, radius) {
    this.pending.grabC = [...center, radius];
    this.pending.grabbing = true;
    this._writeParams();
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setBindGroup(0, this.bg);
    pass.setPipeline(this.pipes.grabSelect);
    pass.dispatchWorkgroups(Math.ceil(this.SOLID_N / 256));
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
    pass.dispatchWorkgroups(Math.ceil(this.SOLID_N / 256));
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

  async _readBuffer(buf, size) {
    const st = this.device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, st, 0, size);
    this.device.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const data = st.getMappedRange().slice(0);
    st.destroy();
    return data;
  }

  // full sim state → JSON-serializable object (Space key downloads this; a
  // saved file can be fed back via window.__sim.loadSnapshot for exact repro)
  async snapshot() {
    const N = this.TOTAL;
    const read = (name, size) => this._readBuffer(this.buf[name], size).then(b64encode);
    return {
      version: 1,
      app: 'slice-splash-gpu',
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
      },
    };
  }

  restore(snap) {
    const q = this.device.queue;
    for (const [name, data] of Object.entries(snap.buffers)) {
      const buf = b64decode(data);
      if (name === 'flags') {
        // clear grab bits — the snapshot may have been taken mid-grab
        const f = new Uint32Array(buf);
        for (let i = 0; i < f.length; i++) f[i] &= ~2;
      }
      q.writeBuffer(this.buf[name], 0, buf);
    }
    this.fluidNext = snap.fluidNext;
  }

  dispose() {
    for (const b of Object.values(this.buf)) b.destroy();
    for (const b of Object.values(this.staging)) b.destroy();
  }

  // count alive constraints (test/debug)
  async readConsAlive() {
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.buf.cons, 0, this.staging.read, 0, this.NUM_CONS * 16);
    this.device.queue.submit([enc.finish()]);
    await this.staging.read.mapAsync(GPUMapMode.READ);
    const u = new Uint32Array(this.staging.read.getMappedRange().slice(0, this.NUM_CONS * 16));
    this.staging.read.unmap();
    let alive = 0;
    for (let c = 0; c < this.NUM_CONS; c++) alive += u[c * 4 + 3];
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
