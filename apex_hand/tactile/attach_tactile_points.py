#!/usr/bin/env python3
"""Attach tactile point arrays to Apex hand URDFs.

Produces two versions per hand:
  - apex_hand_<side>.urdf           (no tactile points)
  - apex_hand_<side>_tactile.urdf   (with tactile sphere links)

Right-hand thumb tip uses right_f0_link4_sensors.urdf; all other attachments
reuse the shared sensor files under tactile/sensors. Sensor angles
are stored as yaw-pitch-roll and converted to URDF roll-pitch-yaw.

Layout expected by this script:

  <package>/urdf/apex_hand_<side>/apex_hand_<side>.urdf           source, and
  <package>/urdf/apex_hand_<side>/apex_hand_<side>_tactile.urdf   output
  <package>/tactile/sensors/*_sensors.urdf                        tactile points
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


BEGIN_MARKER = "<!-- BEGIN GENERATED TACTILE POINTS -->"
END_MARKER = "<!-- END GENERATED TACTILE POINTS -->"

FOUR_FINGERS = ("index", "middle", "ring", "pinky")

WORKSPACE = Path(__file__).resolve().parents[1]
URDF_DIR = WORKSPACE / "urdf"
SENSORS_DIR = WORKSPACE / "tactile" / "sensors"


def hand_dir(side: str) -> Path:
    """Directory holding the URDF and meshes of one hand."""
    return URDF_DIR / f"apex_hand_{side}"


@dataclass(frozen=True)
class SensorPoint:
    index: int
    xyz: str
    rpy: str


@dataclass(frozen=True)
class Attachment:
    parent_link: str
    source_file: str
    group_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Apex hand URDFs with and without tactile points."
    )
    parser.add_argument(
        "--side",
        choices=("left", "right", "both"),
        default="both",
        help="Hand side to process.",
    )
    parser.add_argument(
        "--sensors-dir",
        type=Path,
        default=SENSORS_DIR,
        help="Directory containing *_sensors.urdf files.",
    )
    parser.add_argument(
        "--radius",
        default="0.0008",
        help="Sphere radius for each tactile point, in meters.",
    )
    parser.add_argument(
        "--mass",
        default="0.000001",
        help="Mass for each tactile point link, in kg.",
    )
    parser.add_argument(
        "--rgba",
        default="1 0.05 0.05 1",
        help="Material color for tactile points.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing files.",
    )
    return parser.parse_args()


def load_sensor_points(path: Path) -> list[SensorPoint]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"<\?xml[^>]*\?>", "", text)
    root = ET.fromstring(f"<root>{text}</root>")

    points: list[SensorPoint] = []
    for sensor in root.findall("sensor"):
        name = sensor.attrib.get("name", "")
        match = re.fullmatch(r"sensor_(\d+)", name)
        if not match:
            raise ValueError(f"Unexpected sensor name {name!r} in {path}")
        origin = sensor.find("origin")
        if origin is None:
            raise ValueError(f"Missing origin for {name!r} in {path}")
        xyz = origin.attrib.get("xyz")
        yaw, pitch, roll = origin.attrib.get("rpy", "0 0 0").split()
        if xyz is None:
            raise ValueError(f"Missing xyz for {name!r} in {path}")
        points.append(
            SensorPoint(index=int(match.group(1)), xyz=xyz, rpy=f"{roll} {pitch} {yaw}")
        )

    return sorted(points, key=lambda point: point.index)


def thumb_tip_source(side: str) -> str:
    if side == "right":
        return "right_f0_link4_sensors.urdf"
    return "f0_link4_sensors.urdf"


def build_attachments(side: str) -> list[Attachment]:
    attachments = [
        Attachment(
            parent_link=f"{side}_palm_tactile",
            source_file=f"palm_link_{side}_sensors.urdf",
            group_name="palm",
        )
    ]

    for link_index, tactile_index in ((2, 0), (3, 1), (4, 2)):
        source = (
            thumb_tip_source(side)
            if link_index == 4
            else f"f0_link{link_index}_sensors.urdf"
        )
        attachments.append(
            Attachment(
                parent_link=f"{side}_thumb_tactile{tactile_index}",
                source_file=source,
                group_name=f"thumb_tactile{tactile_index}",
            )
        )

    for finger in FOUR_FINGERS:
        for link_index, tactile_index in ((1, 0), (2, 1), (3, 2)):
            attachments.append(
                Attachment(
                    parent_link=f"{side}_{finger}_tactile{tactile_index}",
                    source_file=f"f1_link{link_index}_sensors.urdf",
                    group_name=f"{finger}_tactile{tactile_index}",
                )
            )

    return attachments


def remove_existing_generated_block(text: str) -> str:
    pattern = re.compile(
        rf"\n?\s*{re.escape(BEGIN_MARKER)}.*?{re.escape(END_MARKER)}\s*\n?",
        re.DOTALL,
    )
    cleaned = pattern.sub("\n", text)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.rstrip() + "\n"


def detect_line_ending(path: Path) -> str:
    data = path.read_bytes()
    return "\r\n" if data.count(b"\r\n") > data.count(b"\n") / 2 else "\n"


def make_point_xml(
    parent_link: str,
    point: SensorPoint,
    radius: str,
    mass: str,
) -> str:
    point_name = f"{parent_link}_point_{point.index:03d}"
    radius_value = float(radius)
    mass_value = float(mass)
    inertia = 0.4 * mass_value * radius_value * radius_value
    inertia_text = f"{inertia:.9g}"
    return f"""  <link name="{point_name}">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <mass value="{mass}" />
      <inertia ixx="{inertia_text}" ixy="0" ixz="0" iyy="{inertia_text}" iyz="0" izz="{inertia_text}" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <sphere radius="{radius}" />
      </geometry>
      <material name="tactile_point_red" />
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <sphere radius="{radius}" />
      </geometry>
    </collision>
  </link>
  <joint name="{point_name}_fix" type="fixed">
    <origin xyz="{point.xyz}" rpy="{point.rpy}" />
    <parent link="{parent_link}" />
    <child link="{point_name}" />
  </joint>
"""


def build_generated_block(
    attachments: list[Attachment],
    sensors_dir: Path,
    radius: str,
    mass: str,
    rgba: str,
) -> tuple[str, dict[str, int]]:
    lines = [
        f"  {BEGIN_MARKER}\n",
        '  <material name="tactile_point_red">\n',
        f'    <color rgba="{rgba}" />\n',
        "  </material>\n",
    ]
    counts: dict[str, int] = {}

    for attachment in attachments:
        source_path = sensors_dir / attachment.source_file
        if not source_path.exists():
            raise FileNotFoundError(f"Missing sensor file: {source_path}")
        points = load_sensor_points(source_path)
        counts[attachment.group_name] = len(points)
        lines.append(
            f"  <!-- {attachment.group_name}: {attachment.source_file} -> {attachment.parent_link} -->\n"
        )
        for point in points:
            lines.append(make_point_xml(attachment.parent_link, point, radius, mass))

    lines.append(f"  {END_MARKER}\n")
    return "".join(lines), counts


def insert_before_robot_close(text: str, block: str) -> str:
    stripped = text.rstrip()
    if not stripped.endswith("</robot>"):
        raise ValueError("Target URDF does not end with </robot>")
    return re.sub(r"\s*</robot>\s*$", f"\n{block}</robot>\n", stripped)


def write_text(path: Path, text: str, line_ending: str) -> None:
    if line_ending != "\n":
        text = text.replace("\n", line_ending)
    path.write_text(text, encoding="utf-8", newline="")


def resolve_source_urdf(side: str) -> Path:
    candidates = [
        hand_dir(side) / f"apex_hand_{side}.urdf",
        hand_dir(side) / f"apex_hand_{side}_tactile.urdf",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No source URDF found for {side}. Tried: {', '.join(str(p) for p in candidates)}"
    )


def process_side(
    side: str,
    sensors_dir: Path,
    radius: str,
    mass: str,
    rgba: str,
    dry_run: bool,
) -> None:
    base_path = hand_dir(side) / f"apex_hand_{side}.urdf"
    tactile_path = hand_dir(side) / f"apex_hand_{side}_tactile.urdf"
    source_path = resolve_source_urdf(side)

    line_ending = detect_line_ending(source_path)
    source_text = source_path.read_text(encoding="utf-8")
    base_text = remove_existing_generated_block(source_text)

    attachments = build_attachments(side)
    existing_links = set(re.findall(r'<link\s+name="([^"]+)"', base_text))
    missing_links = sorted(
        {attachment.parent_link for attachment in attachments} - existing_links
    )
    if missing_links:
        raise SystemExit(
            f"Target URDF is missing parent links for {side}: {', '.join(missing_links)}"
        )

    block, counts = build_generated_block(
        attachments=attachments,
        sensors_dir=sensors_dir,
        radius=radius,
        mass=mass,
        rgba=rgba,
    )
    tactile_text = insert_before_robot_close(base_text, block)
    total = sum(counts.values())

    print(f"Side: {side}")
    print(f"  Source: {source_path}")
    print(f"  Base (no tactile): {base_path}")
    print(f"  Tactile: {tactile_path}")
    print(f"  Generated tactile points: {total}")
    for name, count in counts.items():
        print(f"    {name}: {count}")

    if dry_run:
        return

    write_text(base_path, base_text, line_ending)
    write_text(tactile_path, tactile_text, line_ending)


def main() -> None:
    args = parse_args()
    sides = ("left", "right") if args.side == "both" else (args.side,)

    for side in sides:
        process_side(
            side=side,
            sensors_dir=args.sensors_dir,
            radius=args.radius,
            mass=args.mass,
            rgba=args.rgba,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
