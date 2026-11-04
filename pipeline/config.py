"""Robot description config: everything hardware-specific lives in a YAML file, not in code.

All lengths are in metres, angles in radians unless the field name says otherwise.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict


def _read_yaml(path: Path) -> dict:
    """YAML with `extends: other.yaml`: the other file is loaded first and this one overrides it key by key."""
    raw = yaml.safe_load(path.read_text())
    base = raw.pop("extends", None)
    return _merge(_read_yaml(path.parent / base), raw) if base else raw


def _merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        both_dicts = isinstance(value, dict) and isinstance(merged.get(key), dict)
        merged[key] = _merge(merged[key], value) if both_dicts else value
    return merged


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DiskSpec(_Strict):
    stl: str
    repeat: int = 1


class StaticPart(_Strict):
    """Visual-only part fixed to the world (base platform, walls, ...). Pose is in metres / degrees."""

    stl: str
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rgba: tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0)


class Backbone(_Strict):
    """Elastic tube through the disk centres. Bending stiffness becomes the joint stiffness."""

    outer_diameter: float
    inner_diameter: float = 0.0
    youngs_modulus: float  # Pa
    density: float = 6450.0  # kg/m^3, NiTi


class Joint(_Strict):
    type: Literal["ball"] = "ball"
    stiffness: float | None = None  # N*m/rad. None: derive EI/spacing from the backbone.
    damping: float = 1e-3  # N*m*s/rad
    armature: float = 1e-6  # kg*m^2
    frictionloss: float = 0.0  # N*m, ball-and-socket friction
    range_deg: float = 30.0  # max bend angle per joint
    pivot_offset: float = -1.0  # pivot along the gap to the parent disk (fraction of spacing); -1 = on the parent disk
    # Keep pivots on the planes that carry tendon holes. Mid-gap pivots make every tendon chord shorten
    # with bend (second order), so taut tendons buckle the chain.


class Chain(_Strict):
    disks: list[DiskSpec]  # bottom (fixed to base) to top (tendon termination)
    spacing: float  # disk-to-disk distance
    base_height: float = 0.0  # z of the bottom disk mid-plane
    density: float = 1240.0  # disk material, kg/m^3 (PLA)
    extra_mass_per_disk: float = 0.0  # bearings, inserts, anything not in the STL
    joint: Joint = Joint()
    backbone: Backbone | None = None
    tendon_holes: Literal["auto"] | list[tuple[float, float]] = "auto"  # disk-frame (x, y) when not auto


class Tendons(_Strict):
    pulley_depth: float = 0.03  # pulleys sit this far below the bottom disk
    carriage_radius: float = 0.08  # radial distance of the servo cable attachment
    slack: float = 0.0  # extra cable length at servo angle 0
    limit_timeconst: float = 0.002  # cable elasticity (MuJoCo solref time constant on the length limit)


class Servo(_Strict):
    """Position-controlled servo winding the tendon on a drum. Command = drum angle in rad."""

    drum_radius: float
    gear_ratio: float = 1.0  # motor turns per drum turn
    rotor_inertia: float = 0.0  # kg*m^2 at the motor shaft
    max_torque: float  # N*m at the drum (stall)
    kp: float  # N*m/rad
    kv: float  # N*m*s/rad
    angle_range: tuple[float, float]  # rad; negative releases cable (slack)
    max_speed: float  # rad/s at the drum, enforced by the env


class Sim(_Strict):
    timestep: float = 0.0005
    control_dt: float = 0.02
    integrator: Literal["Euler", "implicitfast", "RK4"] = "implicitfast"


class RobotConfig(_Strict):
    name: str
    stl_dir: Path
    units: Literal["auto", "mm", "m"] = "auto"
    chain: Chain
    tendons: Tendons = Tendons()
    servo: Servo
    sim: Sim = Sim()
    static_parts: list[StaticPart] = []

    @classmethod
    def load(cls, path: str | Path) -> "RobotConfig":
        path = Path(path)
        cfg = cls.model_validate(_read_yaml(path))
        if not cfg.stl_dir.is_absolute():
            cfg.stl_dir = (path.parent / cfg.stl_dir).resolve()
        return cfg

    def disk_sequence(self) -> list[str]:
        return [d.stl for d in self.chain.disks for _ in range(d.repeat)]
