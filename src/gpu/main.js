import * as THREE from 'three';
import { GpuSim } from './sim.js';
import { Renderer } from './render.js';

const $ = (id) => document.getElementById(id);

async function init() {
  if (!navigator.gpu) throw new Error('WebGPU not available in this browser');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter (try chrome://flags → "Unsafe WebGPU")');
  const hasTs = adapter.features.has('timestamp-query');
  const device = await adapter.requestDevice({
    requiredFeatures: hasTs ? ['timestamp-query'] : [],
    requiredLimits: { maxStorageBuffersPerShaderStage: 10 },
  });
  const info = adapter.info || {};
  $('gpu-name').textContent = `${info.vendor || 'gpu'} ${info.architecture || ''}`.trim();
  device.addEventListener('uncapturederror', (e) => {
    console.error('WebGPU uncaptured:', e.error.constructor.name, e.error.message.slice(0, 800));
  });
  device.lost.then((l) => console.error('WebGPU device LOST:', l.reason, l.message));

  const canvas = $('view');
  // scale knobs: ?fluid=262144&dim=24&emit=120
  const q = new URLSearchParams(location.search);
  const MAX_FLUID = Math.min(1 << 20, parseInt(q.get('fluid')) || 65536);
  const SOLID_DIM = Math.min(32, parseInt(q.get('dim')) || 20);
  const TABLE = Math.max(1 << 18,
    1 << Math.ceil(Math.log2(2 * (MAX_FLUID + SOLID_DIM ** 3))));
  const sim = new GpuSim(device, { MAX_FLUID, SOLID_DIM, TABLE });
  const renderer = new Renderer(device, canvas, sim);

  // ---- timestamps ----
  let ts = null;
  if (hasTs) {
    ts = {
      set: device.createQuerySet({ type: 'timestamp', count: 4 }),
      resolve: device.createBuffer({ size: 32, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }),
      staging: device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
      busy: false, compute: 0, render: 0,
    };
  }

  // ---- camera ----
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
  const orbit = { yaw: 0.7, pitch: 0.45, dist: 9, target: new THREE.Vector3(0, 0.5, 0) };
  function updateCamera() {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    const cp = Math.cos(orbit.pitch), sp = Math.sin(orbit.pitch);
    camera.position.set(
      orbit.target.x + orbit.dist * cp * Math.sin(orbit.yaw),
      orbit.target.y + orbit.dist * sp,
      orbit.target.z + orbit.dist * cp * Math.cos(orbit.yaw));
    camera.lookAt(orbit.target);
  }
  updateCamera();

  // ---- environment ----
  const W = sim.o.params;
  renderer.addMesh(new THREE.CircleGeometry(20, 64).rotateX(-Math.PI / 2),
    { color: [0.13, 0.16, 0.2] });
  const basin = renderer.addMesh(new THREE.BoxGeometry(W.wallX * 2 + 0.3, 0.12, W.wallZ * 2 + 0.3),
    { color: [0.2, 0.24, 0.3] });
  basin.matrix.setPosition(0, -0.058, 0);
  for (const [w, h, dd, x, z] of [
    [W.wallX * 2 + 0.3, W.wallH, 0.14, 0, W.wallZ],
    [W.wallX * 2 + 0.3, W.wallH, 0.14, 0, -W.wallZ],
    [0.14, W.wallH, W.wallZ * 2 + 0.3, W.wallX, 0],
    [0.14, W.wallH, W.wallZ * 2 + 0.3, -W.wallX, 0],
  ]) {
    const wall = renderer.addMesh(new THREE.BoxGeometry(w, h, dd),
      { color: [0.75, 0.87, 1.0], alpha: 0.13, kind: 'glass' });
    wall.matrix.setPosition(x, h / 2, z);
  }
  const pole = renderer.addMesh(new THREE.CylinderGeometry(0.04, 0.04, 1.5, 10),
    { color: [0.2, 0.23, 0.27] });
  pole.matrix.setPosition(-3.1, 0.75, -3.1);
  const nozzle = renderer.addMesh(
    new THREE.CylinderGeometry(0.07, 0.1, 0.55, 16).rotateX(Math.PI / 2),
    { color: [0.83, 0.63, 0.09] });
  const hosePos = new THREE.Vector3(-3.1, 1.55, -3.1);

  const knife = renderer.addMesh(new THREE.BoxGeometry(0.03, 0.9, 0.22).translate(0, 0.75, 0),
    { color: [0.93, 0.95, 0.97] });
  knife.visible = false;
  const MAX_TRAIL = 64;
  const trail = renderer.addDynamicMesh((MAX_TRAIL - 1) * 6,
    { color: [0.5, 0.5, 0.55], alpha: 1, unlit: true, kind: 'trail' });
  let trailPts = [];
  function updateTrail() {
    const n = Math.max(0, trailPts.length - 1);
    const v = new Float32Array(n * 18);
    for (let i = 0; i < n; i++) {
      const a = trailPts[i], b = trailPts[i + 1];
      v.set([a.x, 0.02, a.z, a.x, 2.0, a.z, b.x, 0.02, b.z,
             a.x, 2.0, a.z, b.x, 2.0, b.z, b.x, 0.02, b.z], i * 18);
    }
    trail.update(v, n * 6);
  }

  // ---- tools ----
  const TOOLS = ['grab', 'slice', 'hose'];
  let tool = 'grab';
  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  let pointerDown = false;
  let grabDepth = 0;
  let lastGrabPoint = null;
  let prevRayDir = null;
  let spraying = false;
  let sprayPhase = 0;
  const hoseTarget = new THREE.Vector3();

  function setTool(t) {
    tool = t;
    knife.visible = t === 'slice';
    document.querySelectorAll('[data-tool]').forEach((b) =>
      b.classList.toggle('on', b.dataset.tool === t));
    $('hint').textContent = {
      grab: 'Left-drag the cube to move it. Right-drag orbits, wheel zooms.',
      slice: 'Left-drag a stroke across the cube to slice it. Repeatable.',
      hose: 'Hold left mouse to spray water. Aim with the mouse.',
    }[t];
    canvas.style.cursor = t === 'grab' ? 'grab' : 'crosshair';
  }

  function updateRay(e) {
    ndc.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
    ray.setFromCamera(ndc, camera);
  }

  canvas.addEventListener('pointerdown', async (e) => {
    if (e.button === 2) { orbiting = { x: e.clientX, y: e.clientY }; return; }
    if (e.button !== 0) return;
    updateRay(e);
    pointerDown = true;
    const ro = ray.ray.origin, rd = ray.ray.direction;
    if (tool === 'grab') {
      const hit = await sim.pick([ro.x, ro.y, ro.z], [rd.x, rd.y, rd.z]);
      if (!hit || !pointerDown) return;
      grabDepth = hit.t;
      const p = ro.clone().addScaledVector(rd, hit.t);
      lastGrabPoint = p.clone();
      sim.grabAt([p.x, p.y, p.z], 0.3);
      canvas.style.cursor = 'grabbing';
    } else if (tool === 'slice') {
      prevRayDir = rd.clone();
      trailPts = [];
    } else if (tool === 'hose') {
      spraying = true;
    }
  });

  canvas.addEventListener('pointermove', (e) => {
    if (orbiting) {
      orbit.yaw -= (e.clientX - orbiting.x) * 0.005;
      orbit.pitch = Math.min(1.45, Math.max(0.05, orbit.pitch + (e.clientY - orbiting.y) * 0.005));
      orbiting = { x: e.clientX, y: e.clientY };
      updateCamera();
      return;
    }
    updateRay(e);
    const ro = ray.ray.origin, rd = ray.ray.direction;
    const gp = new THREE.Vector3();
    ray.ray.intersectPlane(groundPlane, gp);

    if (tool === 'grab' && pointerDown && lastGrabPoint) {
      const p = ro.clone().addScaledVector(rd, grabDepth);
      p.y = Math.max(p.y, 0.12);
      const d = p.clone().sub(lastGrabPoint);
      const MAX_DRAG = 0.08; // per frame — fast mouse can't teleport-stretch the body
      if (d.length() > MAX_DRAG) d.setLength(MAX_DRAG);
      sim.pending.grabDelta = [d.x, d.y, d.z];
      lastGrabPoint.add(d);
    } else if (tool === 'slice') {
      if (gp) {
        knife.matrix.setPosition(gp);
        if (pointerDown && prevRayDir) {
          sim.queueCut([ro.x, ro.y, ro.z],
            [prevRayDir.x, prevRayDir.y, prevRayDir.z], [rd.x, rd.y, rd.z]);
          prevRayDir.copy(rd);
          trailPts.push(gp.clone());
          if (trailPts.length > MAX_TRAIL) trailPts.shift();
          updateTrail();
        }
      }
    } else if (tool === 'hose' && gp) {
      hoseTarget.copy(gp);
    }
  });

  let orbiting = null;
  addEventListener('pointerup', (e) => {
    if (e.button === 2) { orbiting = null; return; }
    pointerDown = false;
    spraying = false;
    if (lastGrabPoint) { sim.release(); lastGrabPoint = null; }
    prevRayDir = null;
    trailPts = [];
    updateTrail();
    if (tool === 'grab') canvas.style.cursor = 'grab';
  });
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('wheel', (e) => {
    orbit.dist = Math.min(25, Math.max(3, orbit.dist * (1 + e.deltaY * 0.001)));
    updateCamera();
    e.preventDefault();
  }, { passive: false });

  addEventListener('keydown', (e) => {
    if (e.key === '1') setTool('grab');
    if (e.key === '2') setTool('slice');
    if (e.key === '3') setTool('hose');
    if (e.key === 'r' || e.key === 'R') sim.resetSolid();
    if (e.key === 'c' || e.key === 'C') sim.clearFluid();
  });
  document.querySelectorAll('[data-tool]').forEach((b) =>
    b.addEventListener('click', () => setTool(b.dataset.tool)));
  $('reset').addEventListener('click', () => sim.resetSolid());
  $('drain').addEventListener('click', () => sim.clearFluid());
  setTool('grab');
  addEventListener('resize', updateCamera);

  // ---- hose emission ----
  const EMIT_PER_FRAME = Math.min(256, parseInt(q.get('emit')) || 28);
  const NOZZLE_SPEED = 7.5;
  function sprayStep() {
    const dir = hoseTarget.clone().sub(hosePos).normalize();
    const m = new THREE.Matrix4().lookAt(hosePos, hoseTarget, new THREE.Vector3(0, 1, 0));
    nozzle.matrix.copy(m).setPosition(hosePos);
    if (!spraying) return;
    const side = new THREE.Vector3().crossVectors(dir, new THREE.Vector3(0, 1, 0)).normalize();
    const up = new THREE.Vector3().crossVectors(side, dir).normalize();
    const tip = hosePos.clone().addScaledVector(dir, 0.6);
    sim.emit(EMIT_PER_FRAME, [tip.x, tip.y, tip.z], [dir.x, dir.y, dir.z],
      [side.x, side.y, side.z], [up.x, up.y, up.z], NOZZLE_SPEED, sprayPhase);
    sprayPhase = (sprayPhase + EMIT_PER_FRAME * 2.39996) % (Math.PI * 2);
  }

  // ---- main loop ----
  const lightDir = new THREE.Vector3(-0.5, -1, -0.35).normalize();
  let frames = 0, fpsT = 0, last = performance.now();
  let paused = false;

  function frame(now) {
    requestAnimationFrame(frame);
    const dt = Math.min((now - last) / 1000, 0.05);
    last = now;
    if (!paused) {
      sprayStep();
      const enc = device.createCommandEncoder();
      sim.encode(enc, ts);
      renderer.render(enc, camera, lightDir, ts);
      if (ts) enc.resolveQuerySet(ts.set, 0, 4, ts.resolve, 0);
      if (ts && !ts.busy) {
        enc.copyBufferToBuffer(ts.resolve, 0, ts.staging, 0, 32);
      }
      device.queue.submit([enc.finish()]);
      if (ts && !ts.busy) {
        ts.busy = true;
        ts.staging.mapAsync(GPUMapMode.READ).then(() => {
          const t = new BigInt64Array(ts.staging.getMappedRange());
          ts.compute = Number(t[1] - t[0]) / 1e6;
          ts.render = Number(t[3] - t[2]) / 1e6;
          ts.staging.unmap();
          ts.busy = false;
        });
      }
    }
    frames++;
    fpsT += dt;
    if (fpsT > 0.5) {
      const fps = Math.round(frames / fpsT);
      frames = 0; fpsT = 0;
      sim.readActiveCount().then((n) => {
        $('stats').textContent =
          `${fps} fps · ${sim.SOLID_N.toLocaleString()} body + ${n.toLocaleString()} water` +
          (ts ? ` · sim ${ts.compute.toFixed(1)}ms · draw ${ts.render.toFixed(1)}ms` : '');
      });
    }
  }
  requestAnimationFrame(frame);

  // ---- test/tuning hooks ----
  window.__sim = {
    sim, renderer,
    pause: (v) => { paused = v; },
    setParam: (k, v) => { sim.o.params[k] = v; },
    spray: (on) => { spraying = on; hoseTarget.set(1.5, 0, 1.5); },
    async metrics() {
      const { pos, vel, flags } = await sim.readState();
      const N = sim.SOLID_N, T = sim.TOTAL;
      let sMinY = 1e9, sMaxY = -1e9, sBad = 0, sAvgSp = 0;
      for (let i = 0; i < N; i++) {
        const y = pos[i * 4 + 1];
        if (!isFinite(y)) { sBad++; continue; }
        sMinY = Math.min(sMinY, y); sMaxY = Math.max(sMaxY, y);
        sAvgSp += Math.hypot(vel[i * 4], vel[i * 4 + 1], vel[i * 4 + 2]);
      }
      let fN = 0, fBad = 0, fMaxY = -1e9, fAvgSp = 0, fBelow = 0;
      for (let i = N; i < T; i++) {
        if ((flags[i] & 1) === 0) continue;
        const y = pos[i * 4 + 1];
        if (!isFinite(y)) { fBad++; continue; }
        fN++;
        fMaxY = Math.max(fMaxY, y);
        if (y < 0.5) fBelow++;
        fAvgSp += Math.hypot(vel[i * 4], vel[i * 4 + 1], vel[i * 4 + 2]);
      }
      return {
        solid: { minY: sMinY, maxY: sMaxY, bad: sBad, avgSpeed: sAvgSp / N },
        fluid: { n: fN, bad: fBad, maxY: fMaxY, pooledPct: fN ? Math.round(100 * fBelow / fN) : 0, avgSpeed: fN ? fAvgSp / fN : 0 },
        gpu: ts ? { compute: ts.compute, render: ts.render } : null,
      };
    },
    cutLine(x) {
      // vertical slice through the cube at world x, from the camera
      const o = camera.position;
      let prev = null;
      for (let t = -1.4; t <= 1.4; t += 0.1) {
        const target = new THREE.Vector3(x, 0.6 + t, 0);
        const d = target.sub(o).normalize();
        if (prev) sim.queueCut([o.x, o.y, o.z], [prev.x, prev.y, prev.z], [d.x, d.y, d.z]);
        prev = d.clone();
      }
    },
  };
  $('loading').remove();
}

init().catch((e) => {
  document.getElementById('loading').innerHTML =
    `<b>WebGPU init failed:</b> ${e.message}<br><a href="classic.html" style="color:#7ab3ff">→ open the CPU (WebGL) version instead</a>`;
});
