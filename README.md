# Slice & Splash

A FleX-style **unified particle physics** playground: one particle system handles
the soft-body jelly cube, slicing, and the water — and they all interact for free,
because everything is the same kind of particle.

Two implementations share the same algorithms:

| | `index.html` (**GPU**, default) | `classic.html` (CPU) |
|---|---|---|
| Compute | WebGPU compute shaders (WGSL) | JavaScript typed arrays |
| Body | 8,000 particles / ~94k constraints (20³, `?dim=24` → 13.8k) | 343 particles |
| Fluid | 65,536 particles (`?fluid=262144` → 262k) | 2,600 particles |
| Measured (RTX PRO 6000) | 8k body + 33k water: **4.8 ms** · 14k body + 253k water: **19 ms** | ~12 ms at 3k particles |
| Rendering | raw WebGPU, sphere impostors straight from the sim buffer | three.js, marching-cubes metaballs |

## Run

```bash
npm start          # serves on http://localhost:8741
```

WebGPU needs Chrome/Edge (on Linux enable `chrome://flags/#enable-unsafe-webgpu`
and Vulkan if not on by default). The page falls back with a link to the CPU
version if WebGPU is unavailable.

Scale knobs via URL: `?fluid=262144&dim=24&emit=120`

## Controls

| Input | Action |
|---|---|
| `1` / 🖐 Grab | Left-drag the cube (or a sliced piece) |
| `2` / 🔪 Slice | Left-drag a stroke across the cube — repeatable |
| `3` / 🚿 Hose | Hold left mouse to spray, aim with the mouse |
| `R` / `C` | Fresh cube / drain water |
| Right-drag / wheel | Orbit / zoom |

## How it works (GPU)

All particle state lives in GPU storage buffers; the CPU only encodes ~95 compute
dispatches per frame (5 substeps) and feeds a small uniform (tools, emitter).
Everything is **gather-formulated so no float atomics are needed** (WebGPU has none):

- **Spatial hash grid**: count → hierarchical prefix scan (3 dispatches) → scatter,
  rebuilt every substep.
- **Soft body**: XPBD distance constraints solved with Jacobi gather over a static
  CSR adjacency (3 iterations/substep), with compliance, plasticity (permanent
  dents), and overstretch tearing. Both constraint endpoints compute identical
  verdicts, so concurrent writes are benign.
- **Slicing**: each knife stroke is a camera-ray wedge uploaded as uniforms; a
  compute pass flips `alive=0` on every constraint crossing it. No topology
  reallocation — cutting is just flag-flipping.
- **Fluid**: double-density relaxation (Clavet et al.), gather form: density pass,
  then displacement pass reading both particles' pressures. XSPH viscosity per
  substep. Same algorithm and near-same constants as the CPU version.
- **Coupling**: solid↔fluid contacts in one gather pass (each side applies its own
  mass-weighted half). Grabbing makes picked particles kinematic (GPU `pick` pass
  with packed `atomicMin`, ~1 frame readback latency).
- **Rendering**: particles draw directly from the sim's position buffer as
  billboard sphere impostors with per-fragment depth — zero copies, zero readback.

## Test

```bash
node test/drive.js     # CPU version: settle → slice → grab → hose
# GPU version is driven via window.__sim hooks (see src/gpu/main.js) — needs
# headed Chrome on a display for the hardware adapter; headless gets SwiftShader.
```
