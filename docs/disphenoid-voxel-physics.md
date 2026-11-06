# Disphenoid voxels for soft-body physics

## What a disphenoid voxel is

A **disphenoid voxel** is a tetrahedron with this fixed shape:

- **2 long lines (red)** on opposite edges, which never touch. They are perpendicular to each other.
- **4 short lines (green)** connecting the two long lines.
- **Length ratio:** long : short = 2 : √3, or about 1.15 : 1.

**A disphenoid voxel can only ever have one distinct shape.** Other disphenoids exist in geometry, but this term refers only to these exact proportions. Any two voxels can be moved and rotated to overlap perfectly, and each is identical to its mirror image. The voxel appears in 6 orientations in the lattice, but those are rotations of one shape, not different types.

Every face is the same isosceles triangle, with 2 short lines and 1 long line.

## The lattice

Identical voxels fill space with no gaps:

- **Around a long line:** 4 voxels meet at 90° each.
- **Around a short line:** 6 voxels meet at 60° each.
- **Adding a voxel:** each exposed face allows exactly one placement, so the whole lattice follows from a single voxel.

This is the tetrahedral form of the body-centered cubic (BCC) lattice. At each interior point:

| | Touching the point | Owned per point (shared items counted once) |
|---|---|---|
| **Short lines** | 8 | 4 |
| **Long lines** | 6 | 3 |
| **Voxels** | 24 | 6 |

## The idea

Simulate soft bodies as points connected only by length constraints. The voxel's geometry supplies the structure, so wobble comes from the lines trying to return to their starting lengths, with no separate shear or angle parameters. A plain cubic lattice can shear flat when only its edge lengths are constrained, while this tetrahedral lattice resists that.

## Respecting the two line lengths

The physics must keep long and short lines distinct in two ways.

**1. Each line returns to its own starting length.** Each line stores its type, which gives its rest length. A single shared rest length would distort the lattice.

**2. Each line's influence depends on its type.** Long lines should start with about 1/1.15 of a short line's influence (exactly √3/2, or about 0.866). This matches a real rod, whose stiffness is inversely proportional to its length.

The type is stored once, and its influence is applied every time the line is corrected during wobbling:

```text
rest      = [short: √3, long: 2]        # in lattice units
influence = [short: 1.0, long: √3/2]

for each solver iteration:
    for each line (a, b, type):
        error = distance(a, b) - rest[type]
        correction = error * influence[type] * material_stiffness
        move a and b toward each other by correction / 2 each, along the line
```

The √3/2 value is a starting point, not a measured result. It should be tuned against tests, and stiffness and solver iteration count will interact with it.

## Possible benefits

- **Simple material model:** the main behavior comes from one stiffness value and two line types.
- **Compact data:** a regular lattice can derive neighbors from grid position, like a normal voxel grid. Explicit lists are only needed once the structure changes.
- **GPU-friendly:** every point has the same local pattern, and the lines can be split into groups that don't share points, so each group can be solved in parallel.
- **Fracture:** breaking lines or removing voxels could give tearing and cutting without a separate system.
- **Easy mesh binding:** a mesh vertex can follow the 4 corners of its enclosing voxel using barycentric coordinates.
- **Two surface modes:** pure voxels can run inside actual meshes for physics, while metaballs at voxel centers can render water, generated surfaces, and cuts.

## Risks to test

No physics has been built or tested yet. These are the main questions:

- **Inversion:** lengths alone can't tell a voxel from its flipped state, so hard impacts may leave regions stuck inside out. A signed-volume check is the likely safeguard.
- **Limited material range:** length-only lattices can't freely tune how much a material bulges when squeezed, and nearly incompressible materials like rubber are especially hard.
- **Directional bias:** the lattice may respond differently depending on direction, even with adjusted influence.
- **Weak surfaces:** corners and thin parts have fewer connections and may be floppy or flip.

Good first tests are dropping a wobbly cube, twisting a beam, and comparing √3/2 influence with equal influence.

## Prototype results

The builder app, `disphenoid-voxel-builder.html`, covers geometry and skinning, not physics.

- **Thin lines:** a triangle line changes less in width as it rotates (13.4%) than a square line (29.3%).
- **Metaball settings:** ball radius 1.85, threshold 1.50, and detail 2 looked best at the prototype's resolution. These may change on larger objects.
- **Coupled sliders:** tying threshold to radius using a line's width didn't hold across all shapes. Matching the square line made the triangle line disappear, so manual control is still needed.
