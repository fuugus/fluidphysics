# Slice & Splash

A FleX-style **unified particle physics** playground in the browser: one particle
system handles the soft-body jelly cube, slicing, and the water — and they all
interact with each other for free, because everything is the same kind of particle.

## Run

```bash
npm start          # serves on http://localhost:8741
```

(or any static server — `python3 -m http.server 8741` works too)

## Controls

| Input | Action |
|---|---|
| `1` / 🖐 Grab | Left-drag the cube (or a sliced piece) to lift, drag, drop it |
| `2` / 🔪 Slice | Left-drag a stroke across the cube to cut it — repeatable |
| `3` / 🚿 Hose | Hold left mouse to spray water, aim with the mouse |
| `R` | Fresh cube |
| `C` | Drain all water |
| Right-drag / wheel | Orbit / zoom camera |

## How it works

- **Solver** (`src/solver.js`): XPBD position-based dynamics, ~4 substeps/frame.
  - *Soft body*: a 7×7×7 particle lattice joined by distance constraints with
    XPBD compliance (squish), plasticity (permanent dents on hard impacts),
    and overstretch breaking (it can tear).
  - *Slicing*: each knife stroke sweeps a wedge of camera rays; any constraint
    crossing the swept surface is deleted. Connected components are then
    recomputed (union-find) so pieces become independent bodies.
  - *Water*: double-density relaxation (Clavet et al. 2005) — pressure +
    near-pressure on a per-substep pair cache, XSPH viscosity. Pools flat,
    splashes, and pushes the cube around via plain particle contacts.
  - Neighbor search via a hashed uniform grid over typed arrays.
- **Rendering** (`src/main.js`): three.js. The jelly is a marching-cubes
  metaball surface over the solid particles (cut surfaces look right
  automatically, no remeshing). Water is instanced spheres.

## Test

```bash
node test/drive.js   # drives the demo headlessly: settle → slice → grab → hose
```
