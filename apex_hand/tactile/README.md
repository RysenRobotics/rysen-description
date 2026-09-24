# Tactile Data and Processing Scripts

Each hand has **619 tactile points** distributed across 16 tactile pad links. See [../README.md](../README.md) for an overview.

## Tactile Point Distribution

| Tactile pad link | Source sensors file | Point count |
|---|---|---|
| `left_palm_tactile` | `palm_link_left_sensors.urdf` | 89 |
| `right_palm_tactile` | `palm_link_right_sensors.urdf` | 89 |
| `{side}_thumb_tactile0` | `f0_link2_sensors.urdf` | 48 |
| `{side}_thumb_tactile1` | `f0_link3_sensors.urdf` | 32 |
| `left_thumb_tactile2` | `f0_link4_sensors.urdf` | 58 |
| `right_thumb_tactile2` | `right_f0_link4_sensors.urdf` | 58 |
| `{side}_{index,middle,ring,pinky}_tactile0` | `f1_link1_sensors.urdf` | 36 |
| `{side}_{index,middle,ring,pinky}_tactile1` | `f1_link2_sensors.urdf` | 14 |
| `{side}_{index,middle,ring,pinky}_tactile2` | `f1_link3_sensors.urdf` | 48 |

- The four fingers share the same sensors data (not split by side); thumb link4 has separate left/right files; `{side}` is `left`/`right`.
- Total per hand: 89 + 48+32+58 + 4×(36+14+48) = **619**.

## sensors/ Data Format

- **XML fragments** exported from point selection on SolidWorks surfaces: multiple top-level `<sensor name="sensor_N" ...>` elements side by side with no common root node, so standard parsers report "junk after document element". The scripts parse them by stripping the `<?xml?>` declaration, wrapping everything in `<root>`, and then parsing.
- Each file has an **example** `<sensor>` in a comment line at the top; the actual tactile points are `sensor_0` .. `sensor_N-1` (N is the point count in the table above), numbered consecutively with no gaps — do not count the example line in statistics.

## remap.json

Mapping from tactile point numbers to a **2D array layout** (rows and columns correspond to the physical layout of the sensor array), used by the tactile data reader to fetch values by array position.

- Three top-level groups:
  - `palm.left` / `palm.right` — a 14×14 array each (left and right hands are mirror images)
  - `thumb["2"|"3"|"4"]` — 6×8 / 6×6 / 8×9 arrays
  - `finger["1"|"2"|"3"]` — 6×6 / 4×5 / 8×8 arrays (shared by the four fingers)
- Each cell holds a tactile point number; `-1` means an empty slot; the number of real points matches the table above (89 / 48 / 32 / 58 / 36 / 14 / 48).
- The palm is split into left/right; the thumb and the four fingers are not split by side.

## Processing Scripts

Dependencies: `numpy`, `trimesh` (not installed in the dev container; run in an environment that has them).

### attach_tactile_points.py — Tactile Point Attachment

Takes the base URDF as the source, reads the sensors fragments, generates a tactile point sphere link + fixed joint for each point, and outputs `*_tactile.urdf`. Generated blocks are marked with `<!-- BEGIN/END GENERATED TACTILE POINTS -->` comments and replaced as a whole on re-run.

```bash
python3 tactile/attach_tactile_points.py --dry-run    # Print only, don't write files
python3 tactile/attach_tactile_points.py              # Generate tactile URDFs for both hands
```

| Option | Default | Description |
|---|---|---|
| `--side` | `both` | `left` / `right` / `both` |
| `--sensors-dir` | `tactile/sensors` | Sensors fragment directory |
| `--radius` | `0.0008` | Tactile sphere radius (m) |
| `--mass` | `0.000001` | Tactile sphere mass (kg) |
| `--rgba` | `1 0.05 0.05 1` | Tactile sphere color |
| `--dry-run` | off | Print stats only, don't write files |

### correct_tactile_normal.py — Tactile Normal Correction

Aligns the z-axis of each tactile point frame with the surface normal at the nearest point on the host link's mesh (trimesh closest-point query), so that the point's z-axis is the contact normal. The palm group's normals are flipped (z-axis points inward); already-aligned points (dot > 1-1e-8) keep their original values.

```bash
python3 tactile/correct_tactile_normal.py
```

**Warning**: Overwrites `*_tactile.urdf` in place with no backup.
