// MPM water (MLS-MPM, APIC transfers) for the tetravoxel sim.
//
// Water particles live in the sim's own pos/vel/flags buffers (the fluid
// range [FLUID0, TOTAL)); this module only adds a background grid over the
// basin and the per-particle affine matrix C. Each solid substep runs `steps`
// MPM steps: water-water interaction happens entirely through the grid.
// Coupling to the tetravoxel balls stays particle-based (sim's `contacts`
// pass); its push is folded back into the MPM velocity by velocityUpdate.
//
// vel.w of a water particle holds J (volume ratio); emit writes -1 there to
// mark a fresh particle (J = 1, C = 0) because emit can't reach this module's
// C buffer.

export const MPM_DEFAULTS = {
  dx: 0.08,                 // grid cell (m); water particles are spawned dx/2 apart
  min: [-4.0, -0.24, -4.0], // grid box: basin + margin, up to splash height
  max: [4.0, 4.0, 4.0],
  steps: 4,                 // MPM steps per solid substep
  bulk: 150,                // compressibility (wave speed^2): low = soft, squishy
  viscosity: 0.15,          // shear resistance: the "goo"
};

const SCALE = 65536; // fixed-point scale for grid atomics (WebGPU has no float atomics)

const shader = (C) => /* wgsl */ `
const FLUID0: u32 = ${C.FLUID0}u;
const BALL0: u32 = ${C.BALL0}u;
const TOTAL: u32 = ${C.TOTAL}u;
const NX: i32 = ${C.dims[0]};
const NY: i32 = ${C.dims[1]};
const NZ: i32 = ${C.dims[2]};
const NODES: u32 = ${C.dims[0] * C.dims[1] * C.dims[2]}u;
const SCALE: f32 = ${SCALE}.0;

struct MU {
  origin: vec4f,      // grid origin xyz, w = dx
  dt: f32, gravity: f32, bulk: f32, visc: f32,
  wallX: f32, wallZ: f32, wallH: f32, wallT: f32,
  maxSpeed: f32, ballMass: f32, _b: f32, _c: f32,
}

@group(0) @binding(0) var<storage, read_write> pos: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> vel: array<vec4f>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<storage, read_write> aff: array<vec4f>;      // C, 3 columns per water particle
// per node: water [mom xyz, mass] then solid [mom xyz, mass] — the cubes'
// voxel balls are a second material in the same grid
@group(0) @binding(4) var<storage, read_write> gridI: array<atomic<i32>>;
// per node: water (velocity, mass), solid (velocity change from water, mass)
@group(0) @binding(5) var<storage, read_write> gridV: array<vec4f>;
@group(0) @binding(6) var<uniform> U: MU;

fn nodeIdx(c: vec3i) -> u32 { return u32(c.x + NX * (c.y + NY * c.z)); }
fn stencilInside(base: vec3i) -> bool {
  return all(base >= vec3i(0)) && base.x + 2 < NX && base.y + 2 < NY && base.z + 2 < NZ;
}
fn weights(fx: vec3f) -> array<vec3f, 3> {
  return array<vec3f, 3>(
    0.5 * (1.5 - fx) * (1.5 - fx),
    0.75 - (fx - 1.0) * (fx - 1.0),
    0.5 * (fx - 0.5) * (fx - 0.5));
}
fn loadC(k: u32) -> mat3x3f {
  return mat3x3f(aff[k * 3u].xyz, aff[k * 3u + 1u].xyz, aff[k * 3u + 2u].xyz);
}

@compute @workgroup_size(256)
fn mpmClear(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= NODES) { return; }
  for (var k = 0u; k < 8u; k++) { atomicStore(&gridI[g.x * 8u + k], 0); }
}

@compute @workgroup_size(128)
fn mpmP2G(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + FLUID0;
  if (i >= TOTAL || (flags[i] & 1u) == 0u) { return; }
  let dx = U.origin.w;
  let invDx = 1.0 / dx;
  let xg = (pos[i].xyz - U.origin.xyz) * invDx;
  let base = vec3i(floor(xg - 0.5));
  if (!stencilInside(base)) { return; }
  let fx = xg - vec3f(base);
  var w = weights(fx);
  let fresh = vel[i].w < 0.0;
  let J = select(vel[i].w, 1.0, fresh);
  var C = mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0));
  if (!fresh) { C = loadC(i - FLUID0); }
  let I3 = mat3x3f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
  // pressure from volume change + Newtonian viscosity on the strain rate
  let tau = I3 * (U.bulk * J * (J - 1.0)) + (C + transpose(C)) * (U.visc * J);
  let A = tau * (-U.dt * 4.0 * invDx * invDx) + C; // particle mass = volume = 1
  let v = vel[i].xyz;
  for (var a = 0; a < 3; a++) {
  for (var b = 0; b < 3; b++) {
  for (var c = 0; c < 3; c++) {
    let off = vec3i(a, b, c);
    let dpos = (vec3f(off) - fx) * dx;
    let weight = w[a].x * w[b].y * w[c].z;
    let mom = (v + A * dpos) * weight;
    let k = nodeIdx(base + off) * 8u;
    atomicAdd(&gridI[k], i32(round(mom.x * SCALE)));
    atomicAdd(&gridI[k + 1u], i32(round(mom.y * SCALE)));
    atomicAdd(&gridI[k + 2u], i32(round(mom.z * SCALE)));
    atomicAdd(&gridI[k + 3u], i32(round(weight * SCALE)));
  }}}
}

// voxel balls → grid (solid channel): mass + velocity of the cube there
@compute @workgroup_size(128)
fn mpmP2GSolid(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + BALL0;
  if (i >= FLUID0 || (flags[i] & 1u) == 0u) { return; }
  let invDx = 1.0 / U.origin.w;
  let xg = (pos[i].xyz - U.origin.xyz) * invDx;
  let base = vec3i(floor(xg - 0.5));
  if (!stencilInside(base)) { return; }
  let fx = xg - vec3f(base);
  var w = weights(fx);
  let v = vel[i].xyz;
  for (var a = 0; a < 3; a++) {
  for (var b = 0; b < 3; b++) {
  for (var c = 0; c < 3; c++) {
    let off = vec3i(a, b, c);
    let wm = w[a].x * w[b].y * w[c].z * U.ballMass;
    let k = nodeIdx(base + off) * 8u + 4u;
    atomicAdd(&gridI[k], i32(round(v.x * wm * SCALE)));
    atomicAdd(&gridI[k + 1u], i32(round(v.y * wm * SCALE)));
    atomicAdd(&gridI[k + 2u], i32(round(v.z * wm * SCALE)));
    atomicAdd(&gridI[k + 3u], i32(round(wm * SCALE)));
  }}}
}

fn solidMassAt(c: vec3i) -> f32 {
  if (any(c < vec3i(0)) || c.x >= NX || c.y >= NY || c.z >= NZ) { return 0.0; }
  return f32(atomicLoad(&gridI[nodeIdx(c) * 8u + 7u])) / SCALE;
}

@compute @workgroup_size(256)
fn mpmGrid(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= NODES) { return; }
  let m = f32(atomicLoad(&gridI[i * 8u + 3u])) / SCALE;
  let ms = f32(atomicLoad(&gridI[i * 8u + 7u])) / SCALE;
  gridV[i * 2u + 1u] = vec4f(0.0, 0.0, 0.0, ms);
  if (m <= 1e-6) { gridV[i * 2u] = vec4f(0.0); return; }
  var v = vec3f(f32(atomicLoad(&gridI[i * 8u])), f32(atomicLoad(&gridI[i * 8u + 1u])),
                f32(atomicLoad(&gridI[i * 8u + 2u]))) / SCALE / m;
  v.y -= U.dt * U.gravity;
  let c = vec3i(i32(i) % NX, (i32(i) / NX) % NY, i32(i) / (NX * NY));
  // water meets cube: water may not move into the cube faster than the
  // cube moves; both share the normal velocity of their common centre of
  // mass, and the cube keeps the change as its push from the water
  if (ms > 1e-6) {
    let vs = vec3f(f32(atomicLoad(&gridI[i * 8u + 4u])), f32(atomicLoad(&gridI[i * 8u + 5u])),
                   f32(atomicLoad(&gridI[i * 8u + 6u]))) / SCALE / ms;
    let grad = vec3f(solidMassAt(c + vec3i(1, 0, 0)) - solidMassAt(c - vec3i(1, 0, 0)),
                     solidMassAt(c + vec3i(0, 1, 0)) - solidMassAt(c - vec3i(0, 1, 0)),
                     solidMassAt(c + vec3i(0, 0, 1)) - solidMassAt(c - vec3i(0, 0, 1)));
    let gl = length(grad);
    var nrm = vec3f(0.0);
    if (gl > 1e-6) { nrm = grad / gl; }          // points into the cube
    let rel = dot(v - vs, nrm);
    if (gl > 1e-6 && rel > 0.0) {
      let vn = (m * dot(v, nrm) + ms * dot(vs, nrm)) / (m + ms);
      gridV[i * 2u + 1u] = vec4f(nrm * (vn - dot(vs, nrm)), ms);
      v += nrm * (vn - dot(v, nrm));
    }
    // deep inside the cube there is no surface direction: water just moves with it
    if (gl <= 1e-6) { v = vs; }
  }
  let p = U.origin.xyz + vec3f(c) * U.origin.w;
  let dx = U.origin.w;
  // ground: no velocity into the floor, a little drag along it
  if (p.y < dx && v.y < 0.0) { v.y = 0.0; v.x *= 0.98; v.z *= 0.98; }
  // basin walls: thin slabs, block velocity into them from either side
  if (p.y < U.wallH + dx) {
    let band = U.wallT + dx;
    if (abs(abs(p.x) - U.wallX) < band && abs(p.z) < U.wallZ + band) {
      let side = sign(p.x);
      let inner = abs(p.x) < U.wallX;
      if ((inner && v.x * side > 0.0) || (!inner && v.x * side < 0.0)) { v.x = 0.0; }
    }
    if (abs(abs(p.z) - U.wallZ) < band && abs(p.x) < U.wallX + band) {
      let side = sign(p.z);
      let inner = abs(p.z) < U.wallZ;
      if ((inner && v.z * side > 0.0) || (!inner && v.z * side < 0.0)) { v.z = 0.0; }
    }
  }
  gridV[i * 2u] = vec4f(v, m);
}

@compute @workgroup_size(128)
fn mpmG2P(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + FLUID0;
  if (i >= TOTAL || (flags[i] & 1u) == 0u) { return; }
  let dx = U.origin.w;
  let invDx = 1.0 / dx;
  var x = pos[i].xyz;
  let xg = (x - U.origin.xyz) * invDx;
  let base = vec3i(floor(xg - 0.5));
  var J = select(vel[i].w, 1.0, vel[i].w < 0.0);
  var nv = vec3f(0.0);
  var nC = mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0));
  if (stencilInside(base)) {
    let fx = xg - vec3f(base);
    var w = weights(fx);
    for (var a = 0; a < 3; a++) {
    for (var b = 0; b < 3; b++) {
    for (var c = 0; c < 3; c++) {
      let off = vec3i(a, b, c);
      let dpos = vec3f(off) - fx;
      let weight = w[a].x * w[b].y * w[c].z;
      let gv = gridV[nodeIdx(base + off) * 2u].xyz;
      nv += gv * weight;
      nC += mat3x3f(gv * dpos.x, gv * dpos.y, gv * dpos.z) * (4.0 * invDx * weight);
    }}}
    J = clamp(J * (1.0 + U.dt * (nC[0].x + nC[1].y + nC[2].z)), 0.6, 1.4);
  } else {
    // outside the grid (high spray, or spilled past the basin): ballistic
    nv = vel[i].xyz;
    nv.y -= U.dt * U.gravity;
    J = 1.0;
  }
  let sp = length(nv);
  if (sp > U.maxSpeed) { nv *= U.maxSpeed / sp; }
  x += nv * U.dt;
  pos[i] = vec4f(x, pos[i].w);
  vel[i] = vec4f(nv, J);
  let k = (i - FLUID0) * 3u;
  aff[k] = vec4f(nC[0], 0.0);
  aff[k + 1u] = vec4f(nC[1], 0.0);
  aff[k + 2u] = vec4f(nC[2], 0.0);
}
`;

// grid → voxel balls: the velocity change the water gave the cube
const SOLID_G2P = /* wgsl */ `
@compute @workgroup_size(128)
fn mpmG2PSolid(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + BALL0;
  if (i >= FLUID0 || (flags[i] & 1u) == 0u) { return; }
  let invDx = 1.0 / U.origin.w;
  let xg = (pos[i].xyz - U.origin.xyz) * invDx;
  let base = vec3i(floor(xg - 0.5));
  if (!stencilInside(base)) { return; }
  let fx = xg - vec3f(base);
  var w = weights(fx);
  var dv = vec3f(0.0);
  for (var a = 0; a < 3; a++) {
  for (var b = 0; b < 3; b++) {
  for (var c = 0; c < 3; c++) {
    dv += gridV[nodeIdx(base + vec3i(a, b, c)) * 2u + 1u].xyz * (w[a].x * w[b].y * w[c].z);
  }}}
  vel[i] = vec4f(vel[i].xyz + dv, vel[i].w);
}
`;

export class MpmWater {
  constructor(device, sim, opts = {}) {
    this.device = device;
    this.sim = sim;
    this.o = { ...MPM_DEFAULTS, ...opts };
    const { dx, min, max } = this.o;
    this.dims = [0, 1, 2].map((k) => Math.ceil((max[k] - min[k]) / dx) + 1);
    const nodes = this.dims[0] * this.dims[1] * this.dims[2];
    this.nodes = nodes;
    const ST = GPUBufferUsage.STORAGE;
    this.buf = {
      aff: device.createBuffer({ size: sim.o.MAX_FLUID * 48, usage: ST }),
      gridI: device.createBuffer({ size: nodes * 32, usage: ST }),
      gridV: device.createBuffer({ size: nodes * 32, usage: ST }),
      uni: device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    };
    const module = device.createShaderModule({ code: shader({ FLUID0: sim.FLUID0, BALL0: sim.BALL0, TOTAL: sim.TOTAL, dims: this.dims }) + SOLID_G2P });
    const bgl = device.createBindGroupLayout({ entries: [0, 1, 2, 3, 4, 5, 6].map((b) => ({
      binding: b, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: b === 6 ? 'uniform' : b === 2 ? 'read-only-storage' : 'storage' } })) });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
    this.pipes = {};
    for (const k of ['mpmClear', 'mpmP2G', 'mpmP2GSolid', 'mpmGrid', 'mpmG2P', 'mpmG2PSolid'])
      this.pipes[k] = device.createComputePipeline({ layout, compute: { module, entryPoint: k } });
    this.bg = device.createBindGroup({ layout: bgl, entries: [
      sim.buf.pos, sim.buf.vel, sim.buf.flags, this.buf.aff, this.buf.gridI, this.buf.gridV, this.buf.uni,
    ].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    this.uData = new Float32Array(16);
  }

  // a voxel ball's mass in MPM units (a water particle = 1): its volume
  // (2/3 a^3) at the body's density, relative to a water particle's volume
  ballMass() {
    const a = this.sim.o.spacing, dx = this.o.dx;
    const rho = (this.sim.o.solidMass / (4 * a ** 3)) / (this.sim.o.fluidMass / (dx / 2) ** 3);
    return rho * (2 / 3) * a ** 3 / (dx / 2) ** 3;
  }

  writeParams() {
    const p = this.sim.o.params, o = this.o;
    const dt = p.h / o.steps;
    this.uData.set([...o.min, o.dx, dt, -p.gravity, o.bulk, o.viscosity,
      p.wallX, p.wallZ, p.wallH, p.wallT, p.maxSpeed, this.ballMass(), 0, 0]);
    this.device.queue.writeBuffer(this.buf.uni, 0, this.uData);
  }

  // encode one solid substep's worth of MPM steps into an open compute pass
  encode(pass) {
    const nodesWG = Math.ceil(this.nodes / 256);
    const pWG = Math.ceil(this.sim.o.MAX_FLUID / 128);
    const bWG = Math.ceil(this.sim.VOX_N / 128);
    pass.setBindGroup(0, this.bg);
    for (let s = 0; s < this.o.steps; s++) {
      pass.setPipeline(this.pipes.mpmClear); pass.dispatchWorkgroups(nodesWG);
      pass.setPipeline(this.pipes.mpmP2G); pass.dispatchWorkgroups(pWG);
      pass.setPipeline(this.pipes.mpmP2GSolid); pass.dispatchWorkgroups(bWG);
      pass.setPipeline(this.pipes.mpmGrid); pass.dispatchWorkgroups(nodesWG);
      pass.setPipeline(this.pipes.mpmG2P); pass.dispatchWorkgroups(pWG);
      pass.setPipeline(this.pipes.mpmG2PSolid); pass.dispatchWorkgroups(bWG);
    }
  }

  dispose() { for (const b of Object.values(this.buf)) b.destroy(); }
}
