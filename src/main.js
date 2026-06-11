import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { MarchingCubes } from 'three/addons/objects/MarchingCubes.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { Solver } from './solver.js';

// ---------- physics ----------
const solver = new Solver({ maxFluid: 2600 });
solver.addSolidCube(0, 2.0, 0, 7, 0.18);

// ---------- renderer / scene ----------
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x10141c);
scene.fog = new THREE.Fog(0x10141c, 18, 38);
scene.environment = new THREE.PMREMGenerator(renderer)
  .fromScene(new RoomEnvironment(), 0.04).texture;

const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 100);
camera.position.set(5.2, 3.6, 6.4);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.6, 0);
controls.maxPolarAngle = Math.PI * 0.49;
controls.minDistance = 2.5;
controls.maxDistance = 20;
controls.enableDamping = true;
controls.mouseButtons = { LEFT: null, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.ROTATE };

const sun = new THREE.DirectionalLight(0xfff2e0, 2.6);
sun.position.set(6, 9, 4);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = sun.shadow.camera.bottom = -6;
sun.shadow.camera.right = sun.shadow.camera.top = 6;
sun.shadow.bias = -0.0005;
scene.add(sun, new THREE.AmbientLight(0x506080, 1.1));

// ---------- environment meshes ----------
const W = solver.wall;
{
  const ground = new THREE.Mesh(
    new THREE.CircleGeometry(20, 64).rotateX(-Math.PI / 2),
    new THREE.MeshStandardMaterial({ color: 0x232a36, roughness: 0.9 }));
  ground.receiveShadow = true;
  scene.add(ground);

  const basinFloor = new THREE.Mesh(
    new THREE.BoxGeometry(W.x * 2 + 0.3, 0.12, W.z * 2 + 0.3),
    new THREE.MeshStandardMaterial({ color: 0x39424f, roughness: 0.55, metalness: 0.1 }));
  basinFloor.position.y = -0.06 + 0.001;
  basinFloor.receiveShadow = true;
  scene.add(basinFloor);

  const glass = new THREE.MeshPhysicalMaterial({
    color: 0xcfe8ff, roughness: 0.08, metalness: 0,
    transparent: true, opacity: 0.16,
    side: THREE.DoubleSide, depthWrite: false,
  });
  const mkWall = (w, h, d, x, z, ry) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), glass);
    m.position.set(x, h / 2, z);
    m.rotation.y = ry;
    scene.add(m);
  };
  const t = W.halfThick * 2;
  mkWall(W.x * 2 + 0.3, W.height, t, 0, W.z, 0);
  mkWall(W.x * 2 + 0.3, W.height, t, 0, -W.z, 0);
  mkWall(W.z * 2 + 0.3, W.height, t, W.x, 0, Math.PI / 2);
  mkWall(W.z * 2 + 0.3, W.height, t, -W.x, 0, Math.PI / 2);
}

// ---------- jelly cube (metaballs over solid particles) ----------
const jelly = new MarchingCubes(48, new THREE.MeshPhysicalMaterial({
  color: 0xe83a4e, roughness: 0.18, metalness: 0,
  transmission: 0.35, thickness: 0.8, ior: 1.35,
  clearcoat: 0.6, clearcoatRoughness: 0.3,
}), false, false, 300000);
jelly.isolation = 8;
jelly.castShadow = true;
jelly.receiveShadow = true;
scene.add(jelly);

let jellyAsleep = false;
function updateJelly() {
  // skip the (expensive) field rebuild while the jelly is at rest
  const v = solver.vel;
  let maxV2 = 0;
  for (let i = 0; i < solver.solidCount; i++) {
    const i3 = i * 3;
    maxV2 = Math.max(maxV2, v[i3] * v[i3] + v[i3 + 1] * v[i3 + 1] + v[i3 + 2] * v[i3 + 2]);
  }
  if (maxV2 < 0.0025 && jellyAsleep) return;
  jellyAsleep = maxV2 < 0.0025;

  jelly.reset();
  const p = solver.pos, n = solver.solidCount;
  let minX = 1e9, minY = 1e9, minZ = 1e9, maxX = -1e9, maxY = -1e9, maxZ = -1e9;
  for (let i = 0; i < n; i++) {
    const i3 = i * 3;
    minX = Math.min(minX, p[i3]); maxX = Math.max(maxX, p[i3]);
    minY = Math.min(minY, p[i3 + 1]); maxY = Math.max(maxY, p[i3 + 1]);
    minZ = Math.min(minZ, p[i3 + 2]); maxZ = Math.max(maxZ, p[i3 + 2]);
  }
  const margin = 0.4;
  const s = Math.max(maxX - minX, maxY - minY, maxZ - minZ) / 2 + margin;
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2, cz = (minZ + maxZ) / 2;
  jelly.position.set(cx, cy, cz);
  jelly.scale.setScalar(s);

  const subtract = 12;
  const rn = 0.21 / (2 * s);                       // desired world ball radius (smooth skin)
  const strength = (jelly.isolation + subtract) * rn * rn;
  const inv2s = 1 / (2 * s);
  for (let i = 0; i < n; i++) {
    const i3 = i * 3;
    jelly.addBall(
      (p[i3] - cx) * inv2s + 0.5,
      (p[i3 + 1] - cy) * inv2s + 0.5,
      (p[i3 + 2] - cz) * inv2s + 0.5,
      strength, subtract);
  }
  jelly.update();
}

// ---------- water (instanced spheres) ----------
const waterMesh = new THREE.InstancedMesh(
  new THREE.IcosahedronGeometry(solver.fluidRadius * 1.85, 1),
  new THREE.MeshPhysicalMaterial({
    color: 0x3fa9f5, roughness: 0.1, metalness: 0,
    transparent: true, opacity: 0.78, envMapIntensity: 1.4,
  }),
  solver.maxFluid);
waterMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
waterMesh.frustumCulled = false;
scene.add(waterMesh);

const _m = new THREE.Matrix4();
function updateWater() {
  const p = solver.pos;
  let m = 0;
  for (let i = solver.solidCount; i < solver.capacity; i++) {
    if (!solver.active[i]) continue;
    const i3 = i * 3;
    _m.makeTranslation(p[i3], p[i3 + 1], p[i3 + 2]);
    waterMesh.setMatrixAt(m++, _m);
  }
  waterMesh.count = m;
  waterMesh.instanceMatrix.needsUpdate = true;
  return m;
}

// ---------- knife visuals ----------
const knife = new THREE.Group();
{
  const blade = new THREE.Mesh(
    new THREE.BoxGeometry(0.03, 0.9, 0.22),
    new THREE.MeshStandardMaterial({
      color: 0xeef2f8, roughness: 0.12, metalness: 0.9, envMapIntensity: 2.0,
    }));
  blade.position.y = 0.75;
  const handle = new THREE.Mesh(
    new THREE.CylinderGeometry(0.045, 0.045, 0.4, 12),
    new THREE.MeshStandardMaterial({ color: 0x6b4a2f, roughness: 0.6 }));
  handle.position.y = 1.4;
  knife.add(blade, handle);
  knife.visible = false;
  scene.add(knife);
}
const MAX_TRAIL = 64;
const trailGeo = new THREE.BufferGeometry();
trailGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAX_TRAIL * 2 * 3), 3));
trailGeo.setIndex(new THREE.BufferAttribute(new Uint16Array((MAX_TRAIL - 1) * 6), 1));
const trail = new THREE.Mesh(trailGeo, new THREE.MeshBasicMaterial({
  color: 0xffffff, transparent: true, opacity: 0.25,
  side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending,
}));
trail.frustumCulled = false;
scene.add(trail);
let trailPts = [];
function updateTrail() {
  const pos = trailGeo.attributes.position.array;
  const idx = trailGeo.index.array;
  for (let i = 0; i < trailPts.length; i++) {
    const q = trailPts[i];
    pos.set([q.x, 0.02, q.z, q.x, 2.0, q.z], i * 6);
    if (i > 0) {
      const a = (i - 1) * 2;
      idx.set([a, a + 1, a + 2, a + 1, a + 3, a + 2], (i - 1) * 6);
    }
  }
  trailGeo.setDrawRange(0, Math.max(0, (trailPts.length - 1) * 6));
  trailGeo.attributes.position.needsUpdate = true;
  trailGeo.index.needsUpdate = true;
}

// ---------- hose visuals ----------
const hose = new THREE.Group();
{
  const body = new THREE.Mesh(
    new THREE.CylinderGeometry(0.07, 0.1, 0.55, 16).rotateX(Math.PI / 2),
    new THREE.MeshStandardMaterial({ color: 0xd4a017, roughness: 0.3, metalness: 0.9 }));
  body.position.z = 0.27;
  const base = new THREE.Mesh(
    new THREE.SphereGeometry(0.14, 16, 12),
    new THREE.MeshStandardMaterial({ color: 0x333a44, roughness: 0.5, metalness: 0.6 }));
  hose.add(body, base);
  const pole = new THREE.Mesh(
    new THREE.CylinderGeometry(0.04, 0.04, 1.5, 10),
    new THREE.MeshStandardMaterial({ color: 0x333a44, roughness: 0.6 }));
  pole.position.set(-3.1, 0.75, -3.1);
  scene.add(pole);
  hose.position.set(-3.1, 1.55, -3.1);
  scene.add(hose);
}
const NOZZLE_SPEED = 7.5;
let hoseTarget = new THREE.Vector3(0, 0, 0);
let spraying = false;
let sprayPhase = 0;
const _up = new THREE.Vector3(0, 1, 0);
function sprayStep() {
  hose.lookAt(hoseTarget);
  if (!spraying) return;
  const dir = hoseTarget.clone().sub(hose.position).normalize();
  const tip = hose.position.clone().addScaledVector(dir, 0.6);
  // spawn on a rotating cross-section disc so particles never overlap
  const side = new THREE.Vector3().crossVectors(dir, _up).normalize();
  const upN = new THREE.Vector3().crossVectors(side, dir).normalize();
  const spacing = solver.fluidRadius * 2.2;
  for (let k = 0; k < 5; k++) {
    sprayPhase += 2.39996; // golden angle
    const rad = spacing * (0.4 + (k % 2));
    const ox = Math.cos(sprayPhase) * rad, oy = Math.sin(sprayPhase) * rad;
    const jitter = () => (Math.random() - 0.5) * 0.4;
    solver.emitFluid(
      tip.x + side.x * ox + upN.x * oy + dir.x * k * spacing * 0.5,
      tip.y + side.y * ox + upN.y * oy + dir.y * k * spacing * 0.5,
      tip.z + side.z * ox + upN.z * oy + dir.z * k * spacing * 0.5,
      dir.x * NOZZLE_SPEED + jitter(),
      dir.y * NOZZLE_SPEED + jitter(),
      dir.z * NOZZLE_SPEED + jitter());
  }
}

// ---------- input / tools ----------
const TOOLS = { GRAB: 'grab', KNIFE: 'knife', HOSE: 'hose' };
let tool = TOOLS.GRAB;
const raycaster = new THREE.Raycaster();
const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const pointer = new THREE.Vector2();
let pointerDown = false;
let grabDepth = 0;
let grabOffsets = [];     // [{ i, ox, oy, oz }]
let prevRayDir = null;

function setTool(t) {
  tool = t;
  knife.visible = t === TOOLS.KNIFE;
  document.querySelectorAll('[data-tool]').forEach((b) =>
    b.classList.toggle('on', b.dataset.tool === t));
  document.getElementById('hint').textContent = {
    grab: 'Left-drag the cube to lift and drop it. Right-drag orbits the camera.',
    knife: 'Left-drag a stroke across the cube to slice. Slice as often as you like.',
    hose: 'Hold left mouse to spray. Aim by moving the mouse.',
  }[t];
  renderer.domElement.style.cursor = t === TOOLS.GRAB ? 'grab' : 'crosshair';
}

function updateRay(e) {
  pointer.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
}

renderer.domElement.addEventListener('pointerdown', (e) => {
  if (e.button !== 0) return;
  updateRay(e);
  pointerDown = true;
  const ro = raycaster.ray.origin, rd = raycaster.ray.direction;

  if (tool === TOOLS.GRAB) {
    const hit = solver.pickSolid(ro.x, ro.y, ro.z, rd.x, rd.y, rd.z, 0.25);
    if (!hit) return;
    grabDepth = hit.t;
    const hp = ro.clone().addScaledVector(rd, hit.t);
    const g = solver.group[hit.id];
    grabOffsets = [];
    const p = solver.pos;
    for (let i = 0; i < solver.solidCount; i++) {
      if (solver.group[i] !== g) continue;
      const i3 = i * 3;
      const dx = p[i3] - hp.x, dy = p[i3 + 1] - hp.y, dz = p[i3 + 2] - hp.z;
      if (dx * dx + dy * dy + dz * dz < 0.35 * 0.35)
        grabOffsets.push({ i, ox: dx, oy: dy, oz: dz });
    }
    for (const o of grabOffsets)
      solver.grabs.push({ i: o.i, x: hp.x + o.ox, y: hp.y + o.oy, z: hp.z + o.oz });
    renderer.domElement.style.cursor = 'grabbing';
  } else if (tool === TOOLS.KNIFE) {
    prevRayDir = rd.clone();
    trailPts = [];
  }
  // hose handled in sprayStep via `spraying`
  if (tool === TOOLS.HOSE) spraying = true;
});

renderer.domElement.addEventListener('pointermove', (e) => {
  updateRay(e);
  const ro = raycaster.ray.origin, rd = raycaster.ray.direction;
  const gp = new THREE.Vector3();
  raycaster.ray.intersectPlane(groundPlane, gp);

  if (tool === TOOLS.GRAB && pointerDown && grabOffsets.length) {
    const hp = ro.clone().addScaledVector(rd, grabDepth);
    hp.y = Math.max(hp.y, 0.15);
    solver.grabs.length = 0;
    for (const o of grabOffsets)
      solver.grabs.push({ i: o.i, x: hp.x + o.ox, y: hp.y + o.oy, z: hp.z + o.oz });
  } else if (tool === TOOLS.KNIFE) {
    if (gp) {
      knife.position.copy(gp);
      if (trailPts.length) {
        const last = trailPts[trailPts.length - 1];
        knife.lookAt(last.x, gp.y, last.z);
        knife.rotateY(Math.PI / 2);
      }
    }
    if (pointerDown && prevRayDir && gp) {
      solver.cut(ro.x, ro.y, ro.z,
        prevRayDir.x, prevRayDir.y, prevRayDir.z, rd.x, rd.y, rd.z);
      prevRayDir.copy(rd);
      trailPts.push(gp.clone());
      if (trailPts.length > MAX_TRAIL) trailPts.shift();
      updateTrail();
    }
  } else if (tool === TOOLS.HOSE && gp) {
    hoseTarget.copy(gp);
  }
});

addEventListener('pointerup', (e) => {
  if (e.button !== 0) return;
  pointerDown = false;
  spraying = false;
  solver.grabs.length = 0;
  grabOffsets = [];
  prevRayDir = null;
  trailPts = [];
  updateTrail();
  if (tool === TOOLS.GRAB) renderer.domElement.style.cursor = 'grab';
});

addEventListener('keydown', (e) => {
  if (e.key === '1') setTool(TOOLS.GRAB);
  if (e.key === '2') setTool(TOOLS.KNIFE);
  if (e.key === '3') setTool(TOOLS.HOSE);
  if (e.key === 'r' || e.key === 'R') solver.resetSolid();
  if (e.key === 'c' || e.key === 'C') solver.clearFluid();
});

document.querySelectorAll('[data-tool]').forEach((b) =>
  b.addEventListener('click', () => setTool(b.dataset.tool)));
document.getElementById('reset').addEventListener('click', () => solver.resetSolid());
document.getElementById('drain').addEventListener('click', () => solver.clearFluid());
setTool(TOOLS.GRAB);

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// ---------- main loop ----------
const stats = document.getElementById('stats');
let last = performance.now();
let acc = 0;
let fpsTime = 0, fpsFrames = 0, fps = 0, msPhys = 0, msJelly = 0;

function animate(now) {
  requestAnimationFrame(animate);
  const dt = Math.min((now - last) / 1000, 0.05);
  last = now;

  acc += dt;
  let steps = 0;
  const t0 = performance.now();
  while (acc >= 1 / 60 && steps < 3) {
    sprayStep();
    solver.step(1 / 60, 4);
    acc -= 1 / 60;
    steps++;
  }
  if (steps === 3) acc = 0; // don't spiral when tab was hidden
  const t1 = performance.now();

  updateJelly();
  const t2 = performance.now();
  const waterCount = updateWater();

  msPhys += t1 - t0; msJelly += t2 - t1;
  fpsFrames++;
  fpsTime += dt;
  if (fpsTime > 0.5) {
    fps = Math.round(fpsFrames / fpsTime);
    stats.textContent = `${fps} fps · ${waterCount} water · ` +
      `phys ${(msPhys / fpsFrames).toFixed(1)}ms · mesh ${(msJelly / fpsFrames).toFixed(1)}ms`;
    fpsTime = 0; fpsFrames = 0; msPhys = 0; msJelly = 0;
  }

  controls.update();
  renderer.render(scene, camera);
}
requestAnimationFrame(animate);
