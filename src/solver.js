// Unified XPBD particle solver (FleX-style): one particle system for
// soft bodies (lattice + distance constraints) and fluid (free particles
// with cohesion). Slicing = deleting constraints swept by the knife.

export const FLUID = 0;
export const SOLID = 1;

export class Solver {
  constructor({ maxFluid = 2600 } = {}) {
    this.maxFluid = maxFluid;
    this.solidCount = 0;          // set by addSolidCube (call once, before stepping)
    this.capacity = 0;

    // tuning
    this.gravity = -10.0;
    this.solidRadius = 0.09;
    this.fluidRadius = 0.075;
    this.compliance = 2.5e-4;     // soft-body squish (higher = softer)
    this.plasticThreshold = 0.32; // strain beyond which dents become permanent
    this.plasticRate = 0.04;
    this.breakStrain = 2.6;       // constraints tear beyond this stretch factor
    // fluid: double-density relaxation (Clavet et al.)
    this.fluidH = 0.19;           // interaction radius
    this.restDensity = 6.0;
    this.stiffness = 0.002;       // pressure response
    this.nearStiffness = 0.008;   // anti-clustering + surface tension
    this.viscosity = 0.3;         // XSPH velocity smoothing
    this.solidDamping = 1.2;      // 1/s
    this.floorFrictionSolid = 0.4;
    this.floorFrictionFluid = 0.04;
    this.wall = { x: 3.5, z: 3.5, height: 0.92, halfThick: 0.07 };
    this.cullRadius = 7.0;        // fluid escaping this far is recycled
    this.maxSpeed = 14.0;         // keeps overlapping spawns from exploding

    this.grabs = [];              // [{ i, x, y, z }] pulled toward targets each substep
    this.grabStiffness = 0.3;
    this.componentsDirty = false;
  }

  _alloc(n) {
    this.capacity = n;
    this.pos = new Float32Array(3 * n);
    this.prev = new Float32Array(3 * n);
    this.vel = new Float32Array(3 * n);
    this.invMass = new Float32Array(n);
    this.type = new Uint8Array(n);
    this.active = new Uint8Array(n);
    this.group = new Int32Array(n);

    this.cellSize = 0.22;
    this.tableSize = 1 << Math.ceil(Math.log2(2 * n));
    this.cellStart = new Int32Array(this.tableSize + 1);
    this.cellEntries = new Int32Array(n);
    this.fluidNext = 0;

    // fluid pair cache (rebuilt once per substep, reused by 3 passes)
    this.maxPairs = this.maxFluid * 32;
    this.pairI = new Int32Array(this.maxPairs);
    this.pairJ = new Int32Array(this.maxPairs);
    this.pairD = new Float32Array(this.maxPairs);
    this.pairCount = 0;
    this.rho = new Float32Array(n);
    this.rhoNear = new Float32Array(n);
    this.press = new Float32Array(n);
    this.pressNear = new Float32Array(n);
    this.dispX = new Float32Array(n);
    this.dispY = new Float32Array(n);
    this.dispZ = new Float32Array(n);
    this.nbrCnt = new Int32Array(n);
  }

  addSolidCube(cx, cy, cz, n = 7, spacing = 0.18, particleMass = 0.08) {
    this._alloc(n * n * n + this.maxFluid);
    const half = (n - 1) * spacing * 0.5;
    let id = 0;
    for (let ix = 0; ix < n; ix++)
      for (let iy = 0; iy < n; iy++)
        for (let iz = 0; iz < n; iz++) {
          const i3 = id * 3;
          this.pos[i3] = cx - half + ix * spacing;
          this.pos[i3 + 1] = cy - half + iy * spacing;
          this.pos[i3 + 2] = cz - half + iz * spacing;
          this.invMass[id] = 1 / particleMass;
          this.type[id] = SOLID;
          this.active[id] = 1;
          this.group[id] = 0;
          id++;
        }
    this.solidCount = id;

    // distance constraints to all lattice neighbors within sqrt(3)*spacing
    this.consA = [];
    this.consB = [];
    this.consRest = [];
    this.consAlive = [];
    this.pairSet = new Set();
    const idx = (ix, iy, iz) => (ix * n + iy) * n + iz;
    const maxD2 = 3 * spacing * spacing + 1e-6;
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
                this.consA.push(a);
                this.consB.push(b);
                this.consRest.push(Math.sqrt(d2));
                this.consAlive.push(1);
                this.pairSet.add(a * this.capacity + b);
              }

    // snapshot for reset
    this._snapshot = {
      pos: this.pos.slice(0, this.solidCount * 3),
      rest: this.consRest.slice(),
    };
  }

  resetSolid() {
    this.pos.set(this._snapshot.pos);
    this.prev.set(this._snapshot.pos);
    this.vel.fill(0, 0, this.solidCount * 3);
    this.consRest = this._snapshot.rest.slice();
    this.pairSet.clear();
    for (let c = 0; c < this.consA.length; c++) {
      this.consAlive[c] = 1;
      this.pairSet.add(this.consA[c] * this.capacity + this.consB[c]);
    }
    for (let i = 0; i < this.solidCount; i++) this.group[i] = 0;
    this.grabs.length = 0;
  }

  emitFluid(x, y, z, vx, vy, vz, mass = 0.03) {
    const i = this.solidCount + this.fluidNext;
    this.fluidNext = (this.fluidNext + 1) % this.maxFluid;
    const i3 = i * 3;
    this.pos[i3] = x; this.pos[i3 + 1] = y; this.pos[i3 + 2] = z;
    this.prev[i3] = x; this.prev[i3 + 1] = y; this.prev[i3 + 2] = z;
    this.vel[i3] = vx; this.vel[i3 + 1] = vy; this.vel[i3 + 2] = vz;
    this.invMass[i] = 1 / mass;
    this.type[i] = FLUID;
    this.active[i] = 1;
  }

  clearFluid() {
    this.active.fill(0, this.solidCount);
  }

  fluidCount() {
    let m = 0;
    for (let i = this.solidCount; i < this.capacity; i++) m += this.active[i];
    return m;
  }

  // ---- spatial hash ----
  _hash(xi, yi, zi) {
    return (((xi * 92837111) ^ (yi * 689287499) ^ (zi * 283923481)) >>> 0) & (this.tableSize - 1);
  }

  _hashPos(i3) {
    const inv = 1 / this.cellSize;
    return this._hash(
      Math.floor(this.pos[i3] * inv),
      Math.floor(this.pos[i3 + 1] * inv),
      Math.floor(this.pos[i3 + 2] * inv));
  }

  _buildGrid() {
    const cs = this.cellStart;
    cs.fill(0);
    for (let i = 0; i < this.capacity; i++)
      if (this.active[i]) cs[this._hashPos(i * 3)]++;
    let sum = 0;
    for (let k = 0; k < this.tableSize; k++) { sum += cs[k]; cs[k] = sum; }
    cs[this.tableSize] = sum;
    for (let i = 0; i < this.capacity; i++)
      if (this.active[i]) this.cellEntries[--cs[this._hashPos(i * 3)]] = i;
  }

  // ---- main step ----
  step(dt, substeps = 5) {
    const h = dt / substeps;
    const invH = 1 / h;
    const p = this.pos, pv = this.prev, v = this.vel;

    for (let s = 0; s < substeps; s++) {
      // integrate
      for (let i = 0; i < this.capacity; i++) {
        if (!this.active[i]) continue;
        const i3 = i * 3;
        v[i3 + 1] += this.gravity * h;
        pv[i3] = p[i3]; pv[i3 + 1] = p[i3 + 1]; pv[i3 + 2] = p[i3 + 2];
        p[i3] += v[i3] * h;
        p[i3 + 1] += v[i3 + 1] * h;
        p[i3 + 2] += v[i3 + 2] * h;
      }

      // mouse grab
      for (const g of this.grabs) {
        const i3 = g.i * 3, k = this.grabStiffness;
        p[i3] += (g.x - p[i3]) * k;
        p[i3 + 1] += (g.y - p[i3 + 1]) * k;
        p[i3 + 2] += (g.z - p[i3 + 2]) * k;
      }

      this._buildGrid();
      this._solveContacts();
      this._solveFluid();
      this._solveConstraints(h);
      this._solveBounds();

      // velocity update + damping
      const dampS = Math.max(0, 1 - this.solidDamping * h);
      const dampF = Math.max(0, 1 - 0.6 * h);
      for (let i = 0; i < this.capacity; i++) {
        if (!this.active[i]) continue;
        const i3 = i * 3;
        v[i3] = (p[i3] - pv[i3]) * invH;
        v[i3 + 1] = (p[i3 + 1] - pv[i3 + 1]) * invH;
        v[i3 + 2] = (p[i3 + 2] - pv[i3 + 2]) * invH;
        const damp = this.type[i] === SOLID ? dampS : dampF;
        v[i3] *= damp; v[i3 + 1] *= damp; v[i3 + 2] *= damp;
        const sp2 = v[i3] * v[i3] + v[i3 + 1] * v[i3 + 1] + v[i3 + 2] * v[i3 + 2];
        if (sp2 > this.maxSpeed * this.maxSpeed) {
          const f = this.maxSpeed / Math.sqrt(sp2);
          v[i3] *= f; v[i3 + 1] *= f; v[i3 + 2] *= f;
        }
      }

      this._applyViscosity(); // per substep — dissipates pressure oscillation
    }
    if (this.componentsDirty) {
      this._recomputeGroups();
      this.componentsDirty = false;
    }
  }

  // solid-solid and solid-fluid contacts. Iterating solids only finds all
  // such pairs (fluid-fluid is handled by _solveFluid).
  _solveContacts() {
    const p = this.pos, cs = this.cellStart, ce = this.cellEntries;
    const inv = 1 / this.cellSize;
    const rs = this.solidRadius, rf = this.fluidRadius;

    for (let i = 0; i < this.solidCount; i++) {
      if (!this.active[i]) continue;
      const i3 = i * 3;
      const xi = Math.floor(p[i3] * inv), yi = Math.floor(p[i3 + 1] * inv), zi = Math.floor(p[i3 + 2] * inv);
      const wi = this.invMass[i];

      for (let dx = -1; dx <= 1; dx++)
        for (let dy = -1; dy <= 1; dy++)
          for (let dz = -1; dz <= 1; dz++) {
            const hsh = this._hash(xi + dx, yi + dy, zi + dz);
            for (let e = cs[hsh]; e < cs[hsh + 1]; e++) {
              const j = ce[e];
              if (j <= i || !this.active[j]) continue; // fluids all have j > i
              const solidJ = this.type[j] === SOLID;
              if (solidJ && this.pairSet.has(i * this.capacity + j)) continue;
              const j3 = j * 3;
              const ddx = p[i3] - p[j3], ddy = p[i3 + 1] - p[j3 + 1], ddz = p[i3 + 2] - p[j3 + 2];
              const d2 = ddx * ddx + ddy * ddy + ddz * ddz;
              const rsum = rs + (solidJ ? rs : rf);
              if (d2 >= rsum * rsum || d2 < 1e-12) continue;
              const d = Math.sqrt(d2);
              const wj = this.invMass[j];
              const f = (rsum - d) / (d * (wi + wj));
              p[i3] += ddx * f * wi; p[i3 + 1] += ddy * f * wi; p[i3 + 2] += ddz * f * wi;
              p[j3] -= ddx * f * wj; p[j3 + 1] -= ddy * f * wj; p[j3 + 2] -= ddz * f * wj;
            }
          }
    }
  }

  // double-density relaxation (Clavet et al.): pressure makes water spread
  // and pool, near-pressure prevents clustering. Pairs are gathered once
  // and reused for density, displacement, and viscosity.
  _solveFluid() {
    const p = this.pos, cs = this.cellStart, ce = this.cellEntries;
    const inv = 1 / this.cellSize;
    const H = this.fluidH, H2 = H * H;
    const maxPair = 0.05 * H;
    const maxTotal = 0.35 * H;
    const { pairI, pairJ, pairD, rho, rhoNear, press, pressNear, dispX, dispY, dispZ } = this;

    rho.fill(0, this.solidCount);
    rhoNear.fill(0, this.solidCount);
    let np = 0;

    // gather pairs (each once: j > i) + accumulate density on both ends
    for (let i = this.solidCount; i < this.capacity; i++) {
      if (!this.active[i]) continue;
      const i3 = i * 3;
      const xi = Math.floor(p[i3] * inv), yi = Math.floor(p[i3 + 1] * inv), zi = Math.floor(p[i3 + 2] * inv);
      for (let dx = -1; dx <= 1; dx++)
        for (let dy = -1; dy <= 1; dy++)
          for (let dz = -1; dz <= 1; dz++) {
            const hsh = this._hash(xi + dx, yi + dy, zi + dz);
            for (let e = cs[hsh]; e < cs[hsh + 1]; e++) {
              const j = ce[e];
              if (j <= i || !this.active[j] || this.type[j] !== FLUID) continue;
              const j3 = j * 3;
              const ddx = p[j3] - p[i3], ddy = p[j3 + 1] - p[i3 + 1], ddz = p[j3 + 2] - p[i3 + 2];
              const d2 = ddx * ddx + ddy * ddy + ddz * ddz;
              if (d2 >= H2 || np >= this.maxPairs) continue;
              const d = Math.sqrt(d2);
              const q = 1 - d / H;
              pairI[np] = i; pairJ[np] = j; pairD[np] = d; np++;
              const q2 = q * q;
              rho[i] += q2; rho[j] += q2;
              rhoNear[i] += q2 * q; rhoNear[j] += q2 * q;
            }
          }
    }
    this.pairCount = np;

    for (let i = this.solidCount; i < this.capacity; i++) {
      press[i] = this.stiffness * (rho[i] - this.restDensity);
      pressNear[i] = this.nearStiffness * rhoNear[i];
      dispX[i] = 0; dispY[i] = 0; dispZ[i] = 0;
    }

    // symmetric displacement per pair
    for (let k = 0; k < np; k++) {
      const i = pairI[k], j = pairJ[k], d = pairD[k];
      if (d < 1e-9) continue;
      const q = 1 - d / H;
      let D = 0.25 * ((press[i] + press[j]) * q + (pressNear[i] + pressNear[j]) * q * q);
      if (D > maxPair) D = maxPair; else if (D < -maxPair) D = -maxPair;
      const i3 = i * 3, j3 = j * 3;
      const f = D / d;
      const ddx = (p[j3] - p[i3]) * f, ddy = (p[j3 + 1] - p[i3 + 1]) * f, ddz = (p[j3 + 2] - p[i3 + 2]) * f;
      dispX[i] -= ddx; dispY[i] -= ddy; dispZ[i] -= ddz;
      dispX[j] += ddx; dispY[j] += ddy; dispZ[j] += ddz;
    }

    for (let i = this.solidCount; i < this.capacity; i++) {
      if (!this.active[i]) continue;
      let sx = dispX[i], sy = dispY[i], sz = dispZ[i];
      const s2 = sx * sx + sy * sy + sz * sz;
      if (s2 > maxTotal * maxTotal) {
        const f = maxTotal / Math.sqrt(s2);
        sx *= f; sy *= f; sz *= f;
      }
      const i3 = i * 3;
      p[i3] += sx; p[i3 + 1] += sy; p[i3 + 2] += sz;
    }
  }

  _solveConstraints(h) {
    const p = this.pos;
    const alpha = this.compliance / (h * h);
    for (let c = 0; c < this.consA.length; c++) {
      if (!this.consAlive[c]) continue;
      const a = this.consA[c], b = this.consB[c];
      const a3 = a * 3, b3 = b * 3;
      const wa = this.invMass[a], wb = this.invMass[b];
      const w = wa + wb;
      if (w === 0) continue;
      const dx = p[a3] - p[b3], dy = p[a3 + 1] - p[b3 + 1], dz = p[a3 + 2] - p[b3 + 2];
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (d < 1e-9) continue;
      const rest = this.consRest[c];
      if (d > rest * this.breakStrain) { this._killConstraint(c); continue; }
      const C = d - rest;
      const dl = -C / (w + alpha);
      const f = dl / d;
      p[a3] += dx * f * wa; p[a3 + 1] += dy * f * wa; p[a3 + 2] += dz * f * wa;
      p[b3] -= dx * f * wb; p[b3 + 1] -= dy * f * wb; p[b3 + 2] -= dz * f * wb;

      const strain = C / rest;
      if (Math.abs(strain) > this.plasticThreshold)
        this.consRest[c] += C * this.plasticRate;
    }
  }

  _killConstraint(c) {
    this.consAlive[c] = 0;
    this.pairSet.delete(this.consA[c] * this.capacity + this.consB[c]);
    this.componentsDirty = true;
  }

  _solveBounds() {
    const p = this.pos, pv = this.prev;
    const W = this.wall;
    for (let i = 0; i < this.capacity; i++) {
      if (!this.active[i]) continue;
      const i3 = i * 3;
      const solid = this.type[i] === SOLID;
      const r = solid ? this.solidRadius : this.fluidRadius;

      // floor
      if (p[i3 + 1] < r) {
        p[i3 + 1] = r;
        const mu = solid ? this.floorFrictionSolid : this.floorFrictionFluid;
        p[i3] -= (p[i3] - pv[i3]) * mu;
        p[i3 + 2] -= (p[i3 + 2] - pv[i3 + 2]) * mu;
      }

      // basin walls (only below wall height, so things can fly over)
      if (p[i3 + 1] < W.height + r) {
        const lim = r + W.halfThick;
        for (const ax of [0, 2]) {
          const wpos = ax === 0 ? W.x : W.z;
          const q = p[i3 + ax];
          if (Math.abs(Math.abs(q) - wpos) < lim) {
            const side = Math.sign(q) || 1;
            const inside = Math.abs(q) < wpos;
            p[i3 + ax] = inside ? side * (wpos - lim) : side * (wpos + lim);
          }
        }
      }

      // recycle far-flung fluid
      if (!solid) {
        const dx = p[i3], dz = p[i3 + 2];
        if (p[i3 + 1] < -3 || dx * dx + dz * dz > this.cullRadius * this.cullRadius)
          this.active[i] = 0;
      }
    }
  }

  // XSPH: blend each fluid particle's velocity toward its neighborhood
  // average, using the pair cache built by _solveFluid this substep
  _applyViscosity() {
    const v = this.vel;
    const k = this.viscosity;
    const { pairI, pairJ, dispX, dispY, dispZ, nbrCnt } = this;
    const np = this.pairCount;
    dispX.fill(0, this.solidCount);
    dispY.fill(0, this.solidCount);
    dispZ.fill(0, this.solidCount);
    nbrCnt.fill(0, this.solidCount);
    for (let p = 0; p < np; p++) {
      const i = pairI[p], j = pairJ[p];
      const i3 = i * 3, j3 = j * 3;
      dispX[i] += v[j3]; dispY[i] += v[j3 + 1]; dispZ[i] += v[j3 + 2]; nbrCnt[i]++;
      dispX[j] += v[i3]; dispY[j] += v[i3 + 1]; dispZ[j] += v[i3 + 2]; nbrCnt[j]++;
    }
    for (let i = this.solidCount; i < this.capacity; i++) {
      const m = nbrCnt[i];
      if (!m || !this.active[i]) continue;
      const i3 = i * 3;
      v[i3] += k * (dispX[i] / m - v[i3]);
      v[i3 + 1] += k * (dispY[i] / m - v[i3 + 1]);
      v[i3 + 2] += k * (dispZ[i] / m - v[i3 + 2]);
    }
  }

  // ---- knife: kill constraints crossing the wedge swept between two
  // camera rays (origin o, previous dir d1, current dir d2) ----
  cut(ox, oy, oz, d1x, d1y, d1z, d2x, d2y, d2z) {
    // wedge plane normal n = d1 x d2
    const nx = d1y * d2z - d1z * d2y;
    const ny = d1z * d2x - d1x * d2z;
    const nz = d1x * d2y - d1y * d2x;
    const nLen = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (nLen < 1e-8) return 0;
    const p = this.pos;
    let cuts = 0;
    for (let c = 0; c < this.consA.length; c++) {
      if (!this.consAlive[c]) continue;
      const a3 = this.consA[c] * 3, b3 = this.consB[c] * 3;
      const sa = (p[a3] - ox) * nx + (p[a3 + 1] - oy) * ny + (p[a3 + 2] - oz) * nz;
      const sb = (p[b3] - ox) * nx + (p[b3 + 1] - oy) * ny + (p[b3 + 2] - oz) * nz;
      if (sa * sb >= 0) continue;
      // intersection of segment with the plane
      const t = sa / (sa - sb);
      const px = p[a3] + (p[b3] - p[a3]) * t - ox;
      const py = p[a3 + 1] + (p[b3 + 1] - p[a3 + 1]) * t - oy;
      const pz = p[a3 + 2] + (p[b3 + 2] - p[a3 + 2]) * t - oz;
      // inside the wedge between d1 and d2?  (d1 x p)·n >= 0 and (p x d2)·n >= 0
      const c1 = (d1y * pz - d1z * py) * nx + (d1z * px - d1x * pz) * ny + (d1x * py - d1y * px) * nz;
      const c2 = (py * d2z - pz * d2y) * nx + (pz * d2x - px * d2z) * ny + (px * d2y - py * d2x) * nz;
      if (c1 < 0 || c2 < 0) continue;
      this._killConstraint(c);
      cuts++;
    }
    return cuts;
  }

  _recomputeGroups() {
    const n = this.solidCount;
    const parent = new Int32Array(n);
    for (let i = 0; i < n; i++) parent[i] = i;
    const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    for (let c = 0; c < this.consA.length; c++) {
      if (!this.consAlive[c]) continue;
      const ra = find(this.consA[c]), rb = find(this.consB[c]);
      if (ra !== rb) parent[ra] = rb;
    }
    for (let i = 0; i < n; i++) this.group[i] = find(i);
  }

  // nearest solid particle along a ray (for grabbing)
  pickSolid(ox, oy, oz, dx, dy, dz, maxPerp = 0.2) {
    const p = this.pos;
    let best = -1, bestT = Infinity;
    for (let i = 0; i < this.solidCount; i++) {
      const i3 = i * 3;
      const px = p[i3] - ox, py = p[i3 + 1] - oy, pz = p[i3 + 2] - oz;
      const t = px * dx + py * dy + pz * dz;
      if (t < 0 || t >= bestT) continue;
      const qx = px - dx * t, qy = py - dy * t, qz = pz - dz * t;
      if (qx * qx + qy * qy + qz * qz > maxPerp * maxPerp) continue;
      best = i; bestT = t;
    }
    return best < 0 ? null : { id: best, t: bestT };
  }
}
