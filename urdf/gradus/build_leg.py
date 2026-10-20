"""Generate the Gradus ball-socket leg as a URDF.

Geometry is reconstructed from the public Onshape feature tree; see
``urdf/gradus/onshape_params.json`` for the provenance and the extracted
dimensions. The 16 ball-socket links are represented by 16 pitch-revolute
joints (one bending plane) so the two tendons can be solved analytically.

Run:
    python urdf/gradus/build_leg.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARAMS = json.loads((HERE / "onshape_params.json").read_text())["design"]

N_S1 = PARAMS["links_per_section"]
N = 2 * N_S1
PITCH = PARAMS["joint_pitch_m"]
R_S1 = PARAMS["s1_disk_diameter_m"] / 2
R_S2 = PARAMS["s2_disk_diameter_m"] / 2
HOLE_R_S1 = PARAMS["s1_cable_pitch_radius_m"]
HOLE_R_S2 = PARAMS["s2_cable_pitch_radius_m"]
DISK_T = PARAMS["disk_thickness_m"]

MASS_S1 = 0.055
MASS_S2 = 0.040


def link_xml(i: int) -> str:
    s1 = i < N_S1
    radius = R_S1 if s1 else R_S2
    mass = MASS_S1 if s1 else MASS_S2
    color = "0.36 0.55 0.75 1" if s1 else "0.62 0.65 0.68 1"

    # cylinder of the disk + a thin cone/strut down to the next ball
    ixx = mass * (3 * radius**2 + DISK_T**2) / 12
    izz = mass * radius**2 / 2
    return f"""  <link name="disk_{i}">
    <inertial>
      <origin xyz="0 0 {-PITCH / 2:.6f}" rpy="0 0 0"/>
      <mass value="{mass:.6f}"/>
      <inertia ixx="{ixx:.8e}" ixy="0" ixz="0" iyy="{ixx:.8e}" iyz="0" izz="{izz:.8e}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><cylinder radius="{radius:.6f}" length="{DISK_T:.6f}"/></geometry>
      <material name="disk_{i}_mat"><color rgba="{color}"/></material>
    </visual>
    <visual>
      <origin xyz="0 0 {-PITCH / 2:.6f}" rpy="0 0 0"/>
      <geometry><cylinder radius="0.012" length="{PITCH:.6f}"/></geometry>
      <material name="disk_{i}_strut"><color rgba="{color}"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><cylinder radius="{radius:.6f}" length="{DISK_T:.6f}"/></geometry>
    </collision>
  </link>
"""


def joint_xml(i: int) -> str:
    parent = "base" if i == 0 else f"disk_{i - 1}"
    return f"""  <joint name="joint_{i}" type="revolute">
    <origin xyz="0 0 {-PITCH:.6f}" rpy="0 0 0"/>
    <parent link="{parent}"/>
    <child link="disk_{i}"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.75" upper="1.75" effort="500" velocity="10"/>
    <dynamics damping="0.002" friction="0"/>
  </joint>
"""


def build() -> str:
    parts = [
        '<?xml version="1.0"?>',
        '<robot name="gradus_leg">',
        """  <link name="base">
    <inertial>
      <origin xyz="0 0 0.02" rpy="0 0 0"/>
      <mass value="0.2"/>
      <inertia ixx="0.0002" ixy="0" ixz="0" iyy="0.0002" iyz="0" izz="0.00025"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.03" rpy="0 0 0"/>
      <geometry><cylinder radius="0.055" length="0.06"/></geometry>
      <material name="base_mat"><color rgba="0.2 0.2 0.22 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0.03" rpy="0 0 0"/>
      <geometry><cylinder radius="0.055" length="0.06"/></geometry>
    </collision>
  </link>
""",
    ]
    parts += [link_xml(i) for i in range(N)]
    parts += [joint_xml(i) for i in range(N)]
    parts += [
        f"""  <joint name="foot_joint" type="fixed">
    <origin xyz="0 0 {-PITCH / 2:.6f}" rpy="0 0 0"/>
    <parent link="disk_{N - 1}"/>
    <child link="foot"/>
  </joint>
  <link name="foot">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.03"/>
      <inertia ixx="1.0e-5" ixy="0" ixz="0" iyy="1.0e-5" iyz="0" izz="1.0e-5"/>
    </inertial>
    <visual>
      <geometry><sphere radius="0.018"/></geometry>
      <material name="foot_mat"><color rgba="0.1 0.1 0.1 1"/></material>
    </visual>
    <collision>
      <geometry><sphere radius="0.018"/></geometry>
    </collision>
  </link>
</robot>
""",
    ]
    return "\n".join(parts)


if __name__ == "__main__":
    out = HERE / "gradus_leg.urdf"
    out.write_text(build())
    print(f"wrote {out} ({N} links, length {N * PITCH * 1000:.1f} mm)")
