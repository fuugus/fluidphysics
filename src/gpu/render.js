import * as THREE from 'three';

// Minimal raw-WebGPU forward renderer: sphere impostors pulled straight from
// the sim's position buffer (zero copies), plus simple lit meshes for the
// environment. three.js is used only for math and geometry generation.

const IMPOSTOR_WGSL = /* wgsl */ `
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
}
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> pos: array<vec4f>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(1) @binding(0) var<uniform> mat: Mat;

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) uv: vec2f,
  @location(1) viewCenter: vec3f,
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
  let c = mat.color.rgb * (ambient + diff * 0.85) + vec3f(spec) + vec3f(rim * 0.35);

  var out: FOut;
  out.color = vec4f(c, 1.0);
  out.depth = clip.z / clip.w;
  return out;
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
    this.sim = sim;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx = canvas.getContext('webgpu');
    this.ctx.configure({ device, format: this.format, alphaMode: 'opaque' });

    this.camUB = device.createBuffer({ size: 144, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.camData = new Float32Array(36);

    this._buildPipelines();
    this.objects = [];
    this._resize();
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
      ],
    });
    this.matBGL = d.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
    });

    const impMod = d.createShaderModule({ code: IMPOSTOR_WGSL });
    this.impPipe = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.impCamBGL, this.matBGL] }),
      vertex: { module: impMod, entryPoint: 'vs' },
      fragment: { module: impMod, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-strip' },
      depthStencil: { format: depthFmt, depthWriteEnabled: true, depthCompare: 'less' },
    });

    const meshMod = d.createShaderModule({ code: MESH_WGSL });
    const meshDesc = (blend, depthWrite) => ({
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
      depthStencil: { format: depthFmt, depthWriteEnabled: depthWrite, depthCompare: 'less' },
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
    this.trailPipe = d.createRenderPipeline(meshDesc(addBlend, false));

    this.camBG = d.createBindGroup({
      layout: this.camBGL,
      entries: [{ binding: 0, resource: { buffer: this.camUB } }],
    });
    this.impCamBG = d.createBindGroup({
      layout: this.impCamBGL,
      entries: [
        { binding: 0, resource: { buffer: this.camUB } },
        { binding: 1, resource: { buffer: this.sim.buf.pos } },
        { binding: 2, resource: { buffer: this.sim.buf.flags } },
      ],
    });

    // particle material uniforms
    this.materials = {};
    for (const [name, color, radius, base, count, shiny, fresnel] of [
      ['jelly', [0.88, 0.12, 0.2], this.sim.o.params.solidRadius * 2.0, 0, this.sim.SOLID_N, 0.9, 0.5],
      ['water', [0.15, 0.5, 0.92], this.sim.o.params.fluidRadius * 1.8, this.sim.SOLID_N, this.sim.o.MAX_FLUID, 1.2, 0.9],
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
      d.queue.writeBuffer(ub, 0, data);
      this.materials[name] = {
        count,
        bg: d.createBindGroup({ layout: this.matBGL, entries: [{ binding: 0, resource: { buffer: ub } }] }),
      };
    }
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

  render(encoder, camera, lightDir, timestamps = null) {
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

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.ctx.getCurrentTexture().createView(),
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

    const drawMeshes = (kind, pipe) => {
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

    drawMeshes('opaque', this.meshPipe);

    pass.setPipeline(this.impPipe);
    pass.setBindGroup(0, this.impCamBG);
    for (const m of [this.materials.jelly, this.materials.water]) {
      pass.setBindGroup(1, m.bg);
      pass.draw(4, m.count);
    }

    drawMeshes('glass', this.glassPipe);
    drawMeshes('trail', this.trailPipe);
    pass.end();
  }
}
