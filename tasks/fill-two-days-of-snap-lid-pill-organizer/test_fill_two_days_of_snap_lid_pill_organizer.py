"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest fill-two-days-of-snap-lid-pill-organizer    (in an environment with yam_arena installed)
"""

import math
from pathlib import Path

import numpy as np
import pytest
from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.modules.pop("yam_parts", None)
import yam_parts
sys.modules.pop("yam_joints", None)
import yam_joints
from dataclasses import replace

TASK = load(Path(__file__).with_name("fill_two_days_of_snap_lid_pill_organizer.py"))
T = TASK.module
OBJ = {o.name: o for o in T.OBJECTS}
RUN_LIMIT_STEPS = 3000


def _rot(yaw_deg, tilt_deg, axis):
    y, t = math.radians(yaw_deg), math.radians(tilt_deg)
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    c, s = math.cos(t), math.sin(t)
    rt = np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) if axis == "x" else np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return rz @ rt


def _half(o, r):
    if hasattr(o, "size"):
        return np.abs(r) @ (np.asarray(o.size) / 2)
    if hasattr(o, "height"):
        a = r[:, 2]
        return o.height / 2 * np.abs(a) + o.radius * np.sqrt(np.clip(1 - a * a, 0, 1))
    return np.full(3, o.radius)


def place(name, pos, yaw_deg=0.0, tilt_deg=0.0, tilt_axis="x", vel=(0.0, 0.0, 0.0), quat=None, joints=None):
    """The ObjectState of `name` with its centre at `pos`, turned and tipped; its bounding box follows the turn."""
    r = (_rot(yaw_deg, tilt_deg, tilt_axis) if quat is None else
         ik.quat_xyzw_to_matrix(np.asarray(quat, float) / np.linalg.norm(quat)))
    if isinstance(OBJ[name], yam_joints.Mechanism):
        q = {link.name: float((joints or {}).get(link.name, 0.0)) *
             (math.pi / 180 if isinstance(link.joint, yam_joints.Hinge) else 1.0)
             for link in OBJ[name].links}
        state = OBJ[name].offline_state(pos, math.radians(yaw_deg), q, R=r)
        return replace(state, lin_vel=np.asarray(vel, float))
    if isinstance(OBJ[name], yam_parts.Part):
        origin = np.asarray(pos, float)
        low, high = OBJ[name].aabb(origin, r)
        return yam_parts.PartState((low + high) / 2, np.asarray(ik.matrix_to_quat_xyzw(r), float),
                                   np.asarray(vel, float), np.zeros(3), low, high, origin)
    pos, half = np.asarray(pos, float), _half(OBJ[name], r)
    q = ik.matrix_to_quat_xyzw(r)
    return ObjectState(pos, np.asarray(q, float), np.asarray(vel, float), np.zeros(3), pos - half, pos + half)


def layout(poses):
    """Every object's state: the loose ones as given, the fixed ones where the task file puts them."""
    out = {o.name: place(o.name, o.pos, math.degrees(o.yaw)) for o in T.OBJECTS if o.mass is None}
    out.update({name: place(name, **p) for name, p in poses.items()})
    return out


START_POSES = {
    'organizer': {'pos': (0.19, -0.0134, 0.7475), 'joints': {'lid_sun': 0.0, 'catch_sun': 0.0, 'lid_mon': 0.0, 'catch_mon': 0.0, 'lid_tue': 0.0, 'catch_tue': 0.0, 'lid_wed': 0.0, 'catch_wed': 0.0, 'lid_thu': 0.0, 'catch_thu': 0.0, 'lid_fri': 0.0, 'catch_fri': 0.0, 'lid_sat': 0.0, 'catch_sat': 0.0}},
    'caplet_a': {'pos': (0.085, 0.045, 0.751)},
    'caplet_b': {'pos': (0.085, -0.045, 0.751)},
    'caplet_c': {'pos': (0.085, 0, 0.751)},
}
GOAL_POSES = {
    'organizer': {'pos': (0.19, -0.0134, 0.7475), 'joints': {'lid_sun': 0.0, 'catch_sun': 0.0, 'lid_mon': 0.0, 'catch_mon': 0.0, 'lid_tue': 0.0, 'catch_tue': 0.0, 'lid_wed': 0.0, 'catch_wed': 0.0, 'lid_thu': 0.0, 'catch_thu': 0.0, 'lid_fri': 0.0, 'catch_fri': 0.0, 'lid_sat': 0.0, 'catch_sat': 0.0}},
    'caplet_a': {'pos': (0.19, 0.0403, 0.7522)},
    'caplet_b': {'pos': (0.19, -0.0403, 0.7522)},
    'caplet_c': {'pos': (0.085, 0, 0.751)},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)

GOAL_MODULE = {}


def test_the_task_loads_with_its_loose_objects():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {'caplet_a', 'organizer', 'caplet_b', 'caplet_c'}
    assert TASK.docs and TASK.docs.strip()


def test_the_start_scores_nothing():
    assert T.score(START) == 0.0
    assert not T.success(START)


def test_the_goal_scores_full_and_succeeds(monkeypatch):
    module_values = {}
    for name, expr in GOAL_MODULE.items():
        module_values[name] = eval(expr, {**vars(T), **module_values, "math": math, "__builtins__": {}})
    for name, value in module_values.items():
        monkeypatch.setattr(T, name, value)
    assert T.score(GOAL) == 1.0
    assert T.success(GOAL)


def test_success_waits_for_the_objects_to_settle(monkeypatch):
    module_values = {}
    for name, expr in GOAL_MODULE.items():
        module_values[name] = eval(expr, {**vars(T), **module_values, "math": math, "__builtins__": {}})
    for name, value in module_values.items():
        monkeypatch.setattr(T, name, value)
    moving = layout({n: {**p, "vel": (0.05, 0.0, 0.0)} for n, p in GOAL_POSES.items()})
    assert not T.success(moving)

# (name, the layout it starts from, what's moved, the score it should get, module overrides)
CASES = [
    ('untouched start', START_POSES, {
    }, 0, {}),
    ('both days filled, Monday lid left standing open', GOAL_POSES, {
        'organizer': {'pos': (0.19, -0.0134, 0.7475), 'joints': {'lid_sun': 0.0, 'catch_sun': 0.0, 'lid_mon': 110.0, 'catch_mon': 0.0, 'lid_tue': 0.0, 'catch_tue': 0.0, 'lid_wed': 0.0, 'catch_wed': 0.0, 'lid_thu': 0.0, 'catch_thu': 0.0, 'lid_fri': 0.0, 'catch_fri': 0.0, 'lid_sat': 0.0, 'catch_sat': 0.0}},
    }, 0.75, {}),
    ('both days filled, Thursday lid only resting on its catch (not snapped)', GOAL_POSES, {
        'organizer': {'pos': (0.19, -0.0134, 0.7475), 'joints': {'lid_sun': 0.0, 'catch_sun': 0.0, 'lid_mon': 0.0, 'catch_mon': 0.0, 'lid_tue': 0.0, 'catch_tue': 0.0, 'lid_wed': 0.0, 'catch_wed': 0.0, 'lid_thu': 5.0, 'catch_thu': 0.0, 'lid_fri': 0.0, 'catch_fri': 0.0, 'lid_sat': 0.0, 'catch_sat': 0.0}},
    }, 0.75, {}),
    ('both days filled, both lids left open', GOAL_POSES, {
        'organizer': {'pos': (0.19, -0.0134, 0.7475), 'joints': {'lid_sun': 0.0, 'catch_sun': 0.0, 'lid_mon': 110.0, 'catch_mon': 0.0, 'lid_tue': 0.0, 'catch_tue': 0.0, 'lid_wed': 0.0, 'catch_wed': 0.0, 'lid_thu': 110.0, 'catch_thu': 0.0, 'lid_fri': 0.0, 'catch_fri': 0.0, 'lid_sat': 0.0, 'catch_sat': 0.0}},
    }, 0.5, {}),
    ('both days filled and snapped, but the organizer pushed 40 mm off where it lay', GOAL_POSES, {
        'organizer': {'pos': (0.23, -0.0134, 0.7475), 'joints': {'lid_sun': 0.0, 'catch_sun': 0.0, 'lid_mon': 0.0, 'catch_mon': 0.0, 'lid_tue': 0.0, 'catch_tue': 0.0, 'lid_wed': 0.0, 'catch_wed': 0.0, 'lid_thu': 0.0, 'catch_thu': 0.0, 'lid_fri': 0.0, 'catch_fri': 0.0, 'lid_sat': 0.0, 'catch_sat': 0.0}},
        'caplet_a': {'pos': (0.23, 0.0403, 0.7522)},
        'caplet_b': {'pos': (0.23, -0.0403, 0.7522)},
    }, 0.5, {}),
    ('only Monday filled, its lid snapped', START_POSES, {
        'caplet_a': {'pos': (0.19, 0.0403, 0.7522)},
    }, 0.25, {}),
    ('both days filled, the caplets lying 6 mm toward the back of their compartments', GOAL_POSES, {
        'caplet_a': {'pos': (0.196, 0.0403, 0.7522)},
        'caplet_b': {'pos': (0.196, -0.0403, 0.7522)},
    }, 1, {}),
    ('only Monday filled and its lid left open', START_POSES, {
        'organizer': {'pos': (0.19, -0.0134, 0.7475), 'joints': {'lid_sun': 0.0, 'catch_sun': 0.0, 'lid_mon': 110.0, 'catch_mon': 0.0, 'lid_tue': 0.0, 'catch_tue': 0.0, 'lid_wed': 0.0, 'catch_wed': 0.0, 'lid_thu': 0.0, 'catch_thu': 0.0, 'lid_fri': 0.0, 'catch_fri': 0.0, 'lid_sat': 0.0, 'catch_sat': 0.0}},
        'caplet_a': {'pos': (0.19, 0.0403, 0.7522)},
    }, 0, {}),
    ('the spare caplet put in Tuesday as well', GOAL_POSES, {
        'caplet_c': {'pos': (0.19, 0.0135, 0.7522)},
    }, 0, {}),  # a caplet in the wrong day is a medication error
    ('two caplets in Monday', GOAL_POSES, {
        'caplet_c': {'pos': (0.198, 0.0403, 0.7562)},
    }, 0, {}),
    ("Monday's caplet left lying on top of the shut Monday lid", GOAL_POSES, {
        'caplet_a': {'pos': (0.19, 0.0403, 0.7746)},
    }, 0.25, {}),  # only Thursday holds its caplet
    ("Monday's caplet put in Sunday instead", GOAL_POSES, {
        'caplet_a': {'pos': (0.19, 0.0672, 0.7522)},
    }, 0, {}),
    ('everything right but still held (a gripper centre 10 mm above the organizer)', GOAL_POSES, {
    }, 0, {'TEST_GRIPPERS': "{'left': ((ORG_X, ORG_Y, TOP + 0.033), 0.0), 'right': ((0.12, -0.2, TOP + 0.12), 1.0)}"}),
    ('organizer knocked onto its back', GOAL_POSES, {
        'organizer': {'pos': (0.19, -0.0134, 0.7665), 'tilt_deg': 90, 'tilt_axis': 'y', 'joints': {'lid_sun': 0.0, 'catch_sun': 0.0, 'lid_mon': 0.0, 'catch_mon': 0.0, 'lid_tue': 0.0, 'catch_tue': 0.0, 'lid_wed': 0.0, 'catch_wed': 0.0, 'lid_thu': 0.0, 'catch_thu': 0.0, 'lid_fri': 0.0, 'catch_fri': 0.0, 'lid_sat': 0.0, 'catch_sat': 0.0}},
    }, 0, {}),
]


@pytest.mark.parametrize(("name", "base", "moved", "expected", "module"), CASES, ids=[c[0] for c in CASES])
def test_named_layouts(name, base, moved, expected, module, monkeypatch):
    module_values = {}
    for name, expr in module.items():
        module_values[name] = eval(expr, {**vars(T), **module_values, "math": math, "__builtins__": {}})
    for name, value in module_values.items():
        monkeypatch.setattr(T, name, value)
    objects = layout({**base, **moved})
    assert T.score(objects) == pytest.approx(expected), name
    assert bool(T.success(objects)) == (expected == 1.0), name


def test_the_demo_is_reachable_within_the_joint_limits():
    plan = TASK.demo()
    assert set(plan) <= {"left", "right"}
    for side, waypoints in plan.items():
        arm = ArmPlan(waypoints, station.BASES[side])
        steps = round(arm.duration * 20) + 5
        actions = np.array([arm.action7((k + 1) / 20)[0] for k in range(steps)])
        assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1, side
        assert np.all(actions[:, :6] >= ik.LOWER - 1e-6) and np.all(actions[:, :6] <= ik.UPPER + 1e-6), side


def test_the_demo_fits_in_the_run_limit():
    longest = max(ArmPlan(w, station.BASES[s]).duration for s, w in TASK.demo().items())
    assert (longest + TASK.success_hold_s) * 20 < RUN_LIMIT_STEPS * 0.6
