"""Task-local asset extension for the unmodified yam_arena environment.

Only standard-library modules and the native pure-Python Asset spec are imported here.
The simulation implementation is loaded after yam_arena.scene, which the environment
imports only after starting Isaac Sim. Nothing is written to the environment installation.
"""

import importlib.abc
import importlib.machinery
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from yam_arena.task_file import Asset


def _vector(value, length, label):
    out = tuple(float(v) for v in value)
    if len(out) != length or not all(math.isfinite(v) for v in out):
        raise ValueError(f"{label} must contain {length} finite numbers")
    return out


@dataclass(frozen=True)
class Hinge:
    """Passive Z hinge in local coordinates; angles are relative to the reset orientation."""

    pivot: tuple[float, float, float]
    lower_degrees: float = 0.0
    upper_degrees: float = 115.0
    counter_clockwise_only: bool = True

    def _check(self):
        _vector(self.pivot, 3, "hinge.pivot")
        if not -180 < self.lower_degrees <= 0 < self.upper_degrees < 180:
            raise ValueError("hinge limits must satisfy -180 < lower <= 0 < upper < 180")
        if self.counter_clockwise_only and self.lower_degrees != 0:
            raise ValueError("a counter-clockwise hinge must start at lower_degrees=0")


@dataclass(frozen=True)
class LocalUsdAsset(Asset):
    """A native Asset subclass understood by this submission's runtime adapter.

    The native task loader accepts Asset subclasses without modification. Unlike registry
    assets, this spec poses a local USD by its root. Dynamic files must have one root rigid
    body. The inherited asset/xy/extent fields are unused by the custom spawner.
    """

    asset: str = "multimeter_submission_local_usd"
    xy: tuple[float, float] = (0.0, 0.0)
    extent: float = 1.0
    usd_path: str = ""
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    hinge: Hinge | None = None
    collision_filter: tuple[str, ...] = ()
    friction: tuple[float, float] = (1.0, 0.8)

    @property
    def free(self):
        # The native embodiment jitters free objects on seeded resets. A world-mounted
        # spindle must stay with its casing; the runtime still makes its dial dynamic.
        return self.mass is not None and self.hinge is None

    def _check(self):
        super()._check()
        _vector(self.pos, 3, f"{self.name}.pos")
        if any(v <= 0 for v in _vector(self.scale, 3, f"{self.name}.scale")):
            raise ValueError(f"{self.name}.scale must be positive")
        if not math.isfinite(self.yaw):
            raise ValueError(f"{self.name}.yaw must be finite")
        if self.mass is not None and not math.isfinite(self.mass):
            raise ValueError(f"{self.name}.mass must be finite")
        if any(v < 0 for v in _vector(self.friction, 2, f"{self.name}.friction")):
            raise ValueError(f"{self.name}.friction must be nonnegative")
        path = Path(self.usd_path)
        if path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
            raise ValueError(f"{self.name}.usd_path must name a USD file")
        if not path.is_file():
            raise FileNotFoundError(f"Local USD not found: {path}")
        if self.hinge is not None:
            if not isinstance(self.hinge, Hinge) or self.mass is None:
                raise ValueError(f"{self.name}: a hinge requires Hinge configuration and a mass")
            self.hinge._check()


_installed = set()


def _attach(scene_module):
    if id(scene_module) in _installed:
        return
    name = __name__ + "_runtime"
    runtime = sys.modules.get(name)
    if runtime is None:
        path = Path(__file__).with_name("multimeter_runtime.py")
        spec = importlib.util.spec_from_file_location(name, path)
        runtime = importlib.util.module_from_spec(spec)
        sys.modules[name] = runtime
        spec.loader.exec_module(runtime)
    runtime.install(scene_module, LocalUsdAsset)
    _installed.add(id(scene_module))


class _SceneLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        _attach(module)


class _SceneFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "yam_arena.scene":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _SceneLoader(spec.loader)
        return spec


def install():
    """Enable this task's custom spawner in memory; leave native files and other specs alone."""
    scene = sys.modules.get("yam_arena.scene")
    if scene is not None:
        _attach(scene)
    elif not any(isinstance(finder, _SceneFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _SceneFinder())
