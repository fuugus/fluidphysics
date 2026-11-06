import * as THREE from 'three';
import { Skin } from './skin.js';
import { VREC } from './wgsl.js';

// Minimal raw-WebGPU forward renderer: sphere impostors pulled straight from
// the sim's position buffer (zero copies), plus simple lit meshes for the
// environment. three.js is used only for math and geometry generation.

const IMPOSTOR_WGSL = () => /* wgsl */ `
struct Cam {
  view: mat4x4f,
  proj: mat4x4f,
  lightView: vec4f,   // light dir in view space
}
struct Mat {
  color: vec4f,       // rgb + unused
  radius: f32,
  baseIndex: u32,
  shiny: f32,
  fresnel: f32,
  voxColor: u32,      // 1 = colour per voxel from the voxel list
}
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> pos: array<vec4f>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<storage, read> vox: array<u32>;   // voxel records
@group(1) @binding(0) var<uniform> mat: Mat;

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) uv: vec2f,
  @location(1) viewCenter: vec3f,
  @location(2) @interpolate(flat) color: vec3f,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let i = mat.baseIndex + ii;
  var out: VOut;
  if ((flags[i] & 1u) == 0u) {
    out.clip = vec4f(0.0, 0.0, 2.0, 1.0); // clipped away
    return out;
  }
  let corner = vec2f(f32(vi & 1u) * 2.0 - 1.0, f32(vi >> 1u) * 2.0 - 1.0);
  let vc = (cam.view * vec4f(pos[i].xyz, 1.0)).xyz;
  let vp = vc + vec3f(corner * mat.radius, 0.0);
  out.clip = cam.proj * vec4f(vp, 1.0);
  out.uv = corner;
  out.viewCenter = vc;
  out.color = mat.color.rgb;
  if (mat.voxColor != 0u) { out.color = unpack4x8unorm(vox[ii * ${VREC}u + 5u]).rgb; }
  return out;
}

struct FOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
}

@fragment
fn fs(in: VOut) -> FOut {
  let r2 = dot(in.uv, in.uv);
  if (r2 > 1.0) { discard; }
  let nz = sqrt(1.0 - r2);
  let n = vec3f(in.uv, nz); // view-space normal
  let surf = in.viewCenter + n * mat.radius;
  let clip = cam.proj * vec4f(surf, 1.0);

  let l = normalize(-cam.lightView.xyz);
  let diff = max(dot(n, l), 0.0);
  let v = normalize(-surf);
  let h = normalize(l + v);
  let spec = pow(max(dot(n, h), 0.0), 48.0) * mat.shiny;
  let rim = pow(1.0 - max(dot(n, v), 0.0), 2.5) * mat.fresnel;
  let ambient = 0.35 + 0.15 * n.y;
  let c = in.color * (ambient + diff * 0.85) + vec3f(spec) + vec3f(rim * 0.35);

  var out: FOut;
  out.color = vec4f(c, 1.0);
  out.depth = clip.z / clip.w;
  return out;
}
`;

// Tetravoxel surface: every face of every voxel is a candidate triangle
// (vertex_index → voxel, face, corner). A face is drawn only if its voxel is
// alive and the voxel across it is missing or dead — so the outer skin, cut
// faces and torn faces all appear by themselves. Lines are overlaid from
// barycentrics: long lines yellow, short lines green.
const TETRA_WGSL = () => /* wgsl */ `
struct Cam {
  view: mat4x4f,
  proj: mat4x4f,
  lightView: vec4f,
}
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> pos: array<vec4f>;
@group(0) @binding(2) var<storage, read> vox: array<u32>;   // voxel records
@group(0) @binding(3) var<storage, read> nbr: array<i32>;   // face neighbours

const VREC: u32 = ${VREC}u;

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) bary: vec3f,
  @location(1) viewPos: vec3f,
  @location(2) @interpolate(flat) color: vec3f,
  @location(3) @interpolate(flat) nView: vec3f,
  @location(4) @interpolate(flat) longE: vec3f,
}

fn alive(t: u32) -> bool { return bitcast<f32>(vox[t * VREC + 4u]) > 0.0; }
fn isLong(a: u32, b: u32) -> f32 {
  let lo = min(a, b); let hi = max(a, b);
  return select(0.0, 1.0, (lo == 0u && hi == 1u) || (lo == 2u && hi == 3u));
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var out: VOut;
  out.clip = vec4f(0.0, 0.0, 2.0, 1.0); // culled unless visible
  let t = vi / 12u;
  let f = (vi / 3u) % 4u;
  let k = vi % 3u;
  if (!alive(t)) { return out; }
  let n = nbr[t * 4u + f];
  if (n >= 0 && alive(u32(n))) { return out; } // interior face
  // face opposite corner f, wound outward (voxels have positive volume)
  var FACES = array<vec3u, 4>(vec3u(1u, 2u, 3u), vec3u(0u, 3u, 2u), vec3u(0u, 1u, 3u), vec3u(0u, 2u, 1u));
  let L = FACES[f];
  let b = t * VREC;
  let p0 = pos[vox[b + L.x]].xyz;
  let p1 = pos[vox[b + L.y]].xyz;
  let p2 = pos[vox[b + L.z]].xyz;
  var p = p0;
  if (k == 1u) { p = p1; } else if (k == 2u) { p = p2; }
  let vp = (cam.view * vec4f(p, 1.0)).xyz;
  out.clip = cam.proj * vec4f(vp, 1.0);
  out.viewPos = vp;
  out.nView = normalize((cam.view * vec4f(cross(p1 - p0, p2 - p0), 0.0)).xyz);
  out.bary = vec3f(select(0.0, 1.0, k == 0u), select(0.0, 1.0, k == 1u), select(0.0, 1.0, k == 2u));
  // edge opposite triangle corner j joins the other two
  out.longE = vec3f(isLong(L.y, L.z), isLong(L.x, L.z), isLong(L.x, L.y));
  out.color = unpack4x8unorm(vox[b + 5u]).rgb;
  return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  var n = normalize(in.nView);
  let v = normalize(-in.viewPos);
  if (dot(n, v) < 0.0) { n = -n; }
  let l = normalize(-cam.lightView.xyz);
  let diff = max(dot(n, l), 0.0);
  let h = normalize(l + v);
  let spec = pow(max(dot(n, h), 0.0), 32.0) * 0.2;
  var c = in.color * (0.42 + diff * 0.75) + vec3f(spec);
  // ~1px lines on the triangle edges, faded out once a triangle is only a
  // few pixels across (far away the wireframe would drown the faces)
  let fw = fwidth(in.bary);
  let e = smoothstep(vec3f(0.0), fw * 1.1, in.bary);
  var m = e.x; var lg = in.longE.x;
  if (e.y < m) { m = e.y; lg = in.longE.y; }
  if (e.z < m) { m = e.z; lg = in.longE.z; }
  let triPx = 1.0 / max(max(fw.x, fw.y), max(fw.z, 1e-4));
  let show = clamp((triPx - 8.0) / 16.0, 0.0, 1.0);
  let lineCol = select(vec3f(0.12, 0.85, 0.55), vec3f(1.0, 0.82, 0.25), lg > 0.5);
  c = mix(c, lineCol, (1.0 - m) * show);
  return vec4f(c, 1.0);
}
`;

const MESH_WGSL = /* wgsl */ `
struct Cam {
  view: mat4x4f,
  proj: mat4x4f,
  lightView: vec4f,
}
struct Obj {
  model: mat4x4f,
  color: vec4f,      // rgb + alpha
  params: vec4f,     // x: unlit flag
}
@group(0) @binding(0) var<uniform> cam: Cam;
@group(1) @binding(0) var<uniform> obj: Obj;

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) normal: vec3f,
  @location(1) viewPos: vec3f,
}

@vertex
fn vs(@location(0) p: vec3f, @location(1) n: vec3f) -> VOut {
  var out: VOut;
  let world = obj.model * vec4f(p, 1.0);
  let vp = cam.view * world;
  out.clip = cam.proj * vp;
  out.normal = (cam.view * obj.model * vec4f(n, 0.0)).xyz;
  out.viewPos = vp.xyz;
  return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  if (obj.params.x > 0.5) { return obj.color; }
  let v = normalize(-in.viewPos);
  var n = normalize(in.normal);
  if (dot(n, v) < 0.0) { n = -n; } // double-sided
  let l = normalize(-cam.lightView.xyz);
  let diff = max(dot(n, l), 0.0);
  let h = normalize(l + v);
  let spec = pow(max(dot(n, h), 0.0), 32.0) * 0.25;
  let ambient = 0.4 + 0.2 * n.y;
  return vec4f(obj.color.rgb * (ambient + diff * 0.7) + vec3f(spec), obj.color.a);
}
`;

export class Renderer {
  constructor(device, canvas, sim) {
    this.device = device;
    this.canvas = canvas;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx = canvas.getContext('webgpu');
    this.ctx.configure({ device, format: this.format, alphaMode: 'opaque',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC }); // COPY_SRC: capture()

    this.camUB = device.createBuffer({ size: 144, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.camData = new Float32Array(36);

    this._buildPipelines();
    this.objects = [];
    this.attachSim(sim);
    this._resize();
    this.skin = new Skin(device, sim, this.format);
  }

  // (re)bind particle buffers + materials; called again after a sim rebuild
  attachSim(sim) {
    const d = this.device;
    this.sim = sim;
    if (this.skin) this.skin.attachSim(sim);
    this.impCamBG = d.createBindGroup({
      layout: this.impCamBGL,
      entries: [
        { binding: 0, resource: { buffer: this.camUB } },
        { binding: 1, resource: { buffer: sim.buf.pos } },
        { binding: 2, resource: { buffer: sim.buf.flags } },
        { binding: 3, resource: { buffer: sim.buf.gridS, offset: sim.voxRecOffset, size: sim.voxRecSize } },
      ],
    });
    if (this.nbrBuf) this.nbrBuf.destroy();
    this.nbrBuf = d.createBuffer({ size: sim.host.faceNbr.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.nbrBuf, 0, sim.host.faceNbr);
    this.tetraBG = d.createBindGroup({
      layout: this.tetraBGL,
      entries: [
        { binding: 0, resource: { buffer: this.camUB } },
        { binding: 1, resource: { buffer: sim.buf.pos } },
        { binding: 2, resource: { buffer: sim.buf.gridS, offset: sim.voxRecOffset, size: sim.voxRecSize } },
        { binding: 3, resource: { buffer: this.nbrBuf } },
      ],
    });
    if (this.materials) for (const m of Object.values(this.materials)) m.ub.destroy();
    this.materials = {};
    // one draw for every voxel ball (colour per voxel), one for the water
    for (const [name, color, radius, base, count, shiny, fresnel, voxColor] of [
      ['voxels', [1, 1, 1], sim.o.params.solidRadius, sim.BALL0, sim.VOX_N, 0.9, 0.5, 1],
      ['water', [0.15, 0.5, 0.92], sim.o.params.fluidRadius * 1.8, sim.FLUID0, sim.o.MAX_FLUID, 1.2, 0.9, 0],
    ]) {
      const ub = d.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const data = new ArrayBuffer(48);
      const f = new Float32Array(data);
      const u = new Uint32Array(data);
      f.set([...color, 1], 0);
      f[4] = radius;
      u[5] = base;
      f[6] = shiny;
      f[7] = fresnel;
      u[8] = voxColor;
      d.queue.writeBuffer(ub, 0, data);
      this.materials[name] = {
        count, ub,
        bg: d.createBindGroup({ layout: this.matBGL, entries: [{ binding: 0, resource: { buffer: ub } }] }),
      };
    }
  }

  _buildPipelines() {
    const d = this.device;
    const depthFmt = 'depth24plus';

    // group 0: camera (+ particle buffers for impostor)
    this.camBGL = d.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
    });
    this.impCamBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    this.matBGL = d.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
    });

    const impMod = d.createShaderModule({ code: IMPOSTOR_WGSL() });
    this.impPipe = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.impCamBGL, this.matBGL] }),
      vertex: { module: impMod, entryPoint: 'vs' },
      fragment: { module: impMod, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-strip' },
      depthStencil: { format: depthFmt, depthWriteEnabled: true, depthCompare: 'less' },
    });

    this.tetraBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    const tetraMod = d.createShaderModule({ code: TETRA_WGSL() });
    this.tetraPipe = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.tetraBGL] }),
      vertex: { module: tetraMod, entryPoint: 'vs' },
      fragment: { module: tetraMod, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: depthFmt, depthWriteEnabled: true, depthCompare: 'less' },
    });

    const meshMod = d.createShaderModule({ code: MESH_WGSL });
    const meshDesc = (blend, depthWrite, depthCompare = 'less') => ({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.camBGL, this.matBGL] }),
      vertex: {
        module: meshMod, entryPoint: 'vs',
        buffers: [
          { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] },
          { arrayStride: 12, attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }] },
        ],
      },
      fragment: { module: meshMod, entryPoint: 'fs', targets: [{ format: this.format, ...(blend && { blend }) }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: depthFmt, depthWriteEnabled: depthWrite, depthCompare },
    });
    this.meshPipe = d.createRenderPipeline(meshDesc(null, true));
    const alphaBlend = {
      color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
    };
    this.glassPipe = d.createRenderPipeline(meshDesc(alphaBlend, false));
    const addBlend = {
      color: { srcFactor: 'one', dstFactor: 'one' },
      alpha: { srcFactor: 'one', dstFactor: 'one' },
    };
    this.trailPipe = d.createRenderPipeline(meshDesc(addBlend, false, 'always')); // helper overlay

    this.camBG = d.createBindGroup({
      layout: this.camBGL,
      entries: [{ binding: 0, resource: { buffer: this.camUB } }],
    });
  }

  // ---- environment objects from three.js geometries ----
  addMesh(geometry, { color = [1, 1, 1], alpha = 1, unlit = false, kind = 'opaque' } = {}) {
    const d = this.device;
    const g = geometry.index ? geometry.toNonIndexed() : geometry;
    const posArr = g.attributes.position.array;
    const nrmArr = g.attributes.normal ? g.attributes.normal.array : new Float32Array(posArr.length);
    const vb = d.createBuffer({ size: posArr.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    const nb = d.createBuffer({ size: nrmArr.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(vb, 0, posArr);
    d.queue.writeBuffer(nb, 0, nrmArr);
    const ub = d.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const obj = {
      vb, nb, ub, count: posArr.length / 3, kind, visible: true,
      matrix: new THREE.Matrix4(),
      color: [...color, alpha],
      unlit: unlit ? 1 : 0,
      bg: d.createBindGroup({ layout: this.matBGL, entries: [{ binding: 0, resource: { buffer: ub } }] }),
      uData: new Float32Array(24),
    };
    this.objects.push(obj);
    return obj;
  }

  // dynamic mesh whose vertices are rewritten per frame (knife trail)
  addDynamicMesh(maxVerts, opts) {
    const obj = this.addMesh(new THREE.BufferGeometry().setAttribute('position',
      new THREE.BufferAttribute(new Float32Array(maxVerts * 3), 3)), opts);
    obj.count = 0;
    obj.update = (verts, count) => {
      this.device.queue.writeBuffer(obj.vb, 0, verts, 0, count * 3);
      obj.count = count;
    };
    return obj;
  }

  _resize() {
    const dpr = Math.min(devicePixelRatio, 2);
    const w = Math.floor(this.canvas.clientWidth * dpr) || 8;
    const h = Math.floor(this.canvas.clientHeight * dpr) || 8;
    if (this.canvas.width === w && this.canvas.height === h && this.depthTex) return;
    this.canvas.width = w;
    this.canvas.height = h;
    this.depthTex = this.device.createTexture({
      size: [w, h], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  // debug: PNG data URL of the next presented frame (test hook — Playwright
  // screenshots don't see the WebGPU canvas here)
  capture() { return new Promise((res) => { this._capReq = res; }); }
  _encodeCapture(encoder, tex) {
    const res = this._capReq;
    this._capReq = null;
    const w = tex.width, h = tex.height, bpr = Math.ceil((w * 4) / 256) * 256;
    const buf = this.device.createBuffer({ size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    // encoded first in the frame's encoder, so copy what the frame draws: defer
    // the copy until the render pass is recorded (end of this call)
    this._capAfter = () => encoder.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, { width: w, height: h });
    setTimeout(async () => { // runs after the frame's submit
      await buf.mapAsync(GPUMapMode.READ);
      const src = new Uint8Array(buf.getMappedRange());
      const cv = document.createElement('canvas');
      cv.width = w; cv.height = h;
      const ctx = cv.getContext('2d');
      const img = ctx.createImageData(w, h);
      const bgra = this.format.startsWith('bgra');
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const s0 = y * bpr + x * 4, d0 = (y * w + x) * 4;
        img.data[d0] = src[s0 + (bgra ? 2 : 0)]; img.data[d0 + 1] = src[s0 + 1];
        img.data[d0 + 2] = src[s0 + (bgra ? 0 : 2)]; img.data[d0 + 3] = 255;
      }
      buf.unmap(); buf.destroy();
      ctx.putImageData(img, 0, 0);
      res(cv.toDataURL('image/png'));
    }, 0);
  }

  render(encoder, camera, lightDir, timestamps = null, mode = 'spheres', iso = 1.0, texMode = 0) {
    this._resize();
    // camera uniform
    camera.updateMatrixWorld();
    const view = camera.matrixWorldInverse;
    this.camData.set(view.elements, 0);
    this.camData.set(camera.projectionMatrix.elements, 16);
    const lv = lightDir.clone().transformDirection(view);
    this.camData.set([lv.x, lv.y, lv.z, 0], 32);
    this.device.queue.writeBuffer(this.camUB, 0, this.camData);

    for (const o of this.objects) {
      if (!o.visible) continue;
      o.uData.set(o.matrix.elements, 0);
      o.uData.set(o.color, 16);
      o.uData[20] = o.unlit;
      this.device.queue.writeBuffer(o.ub, 0, o.uData);
    }

    // draws environment meshes of a given kind into an arbitrary render pass
    const drawMeshesTo = (pass, kind, pipe) => {
      pass.setPipeline(pipe);
      pass.setBindGroup(0, this.camBG);
      for (const o of this.objects) {
        if (o.kind !== kind || !o.visible || o.count === 0) continue;
        pass.setBindGroup(1, o.bg);
        pass.setVertexBuffer(0, o.vb);
        pass.setVertexBuffer(1, o.nb);
        pass.draw(o.count);
      }
    };

    const swapTex = this.ctx.getCurrentTexture();
    const swapView = swapTex.createView();
    if (this._capReq) this._encodeCapture(encoder, swapTex);

    // ----- skinned modes: hand off to the multi-pass surface renderer -----
    if (mode !== 'spheres' && mode !== 'tetra') {
      const hq = mode === 'raymarchHQ';
      this.skin.resize(this.canvas.width, this.canvas.height, hq ? 2 : 1);
      this.skin.updateUniforms(camera, lightDir, iso,
        { hq, texMode, waterOnly: mode === 'embed' });
      this.skin.render(encoder, swapView, mode,
        (pass) => drawMeshesTo(pass, 'opaque', this.meshPipe),
        (pass) => {
          drawMeshesTo(pass, 'glass', this.glassPipe);
          drawMeshesTo(pass, 'trail', this.trailPipe);
        });
      if (this._capAfter) { this._capAfter(); this._capAfter = null; }
      return;
    }

    // ----- sphere impostor mode (original direct path) -----
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: swapView,
        clearValue: { r: 0.063, g: 0.078, b: 0.11, a: 1 },
        loadOp: 'clear', storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: this.depthTex.createView(),
        depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store',
      },
      ...(timestamps && {
        timestampWrites: { querySet: timestamps.set, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 },
      }),
    });

    drawMeshesTo(pass, 'opaque', this.meshPipe);

    if (mode === 'tetra') {
      pass.setPipeline(this.tetraPipe);
      pass.setBindGroup(0, this.tetraBG);
      pass.draw(this.sim.VOX_N * 12);
    }

    pass.setPipeline(this.impPipe);
    pass.setBindGroup(0, this.impCamBG);
    for (const [name, m] of Object.entries(this.materials)) {
      if (m.count === 0 || (mode === 'tetra' && name === 'voxels')) continue;
      pass.setBindGroup(1, m.bg);
      pass.draw(4, m.count);
    }

    drawMeshesTo(pass, 'glass', this.glassPipe);
    drawMeshesTo(pass, 'trail', this.trailPipe);
    pass.end();
    if (this._capAfter) { this._capAfter(); this._capAfter = null; }
  }
}
