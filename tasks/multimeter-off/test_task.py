"""The challenge task's scoring, reachable CCW plan, and local-USD loading (no simulator)."""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load

TASK = Path(__file__).resolve().with_name("my_task.py")
ADDON_NAME = load(TASK).module._addon_name


def states(task, degrees, spin=0.0, displacement=(0.0, 0.0, 0.0)):
    pos = task.module.PIVOT_WORLD + np.asarray(displacement)
    yaw = task.module.YAW + math.radians(degrees)
    return {"dial": ObjectState(
        pos, np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]),
        np.zeros(3), np.array([0.0, 0.0, spin]), pos - 0.02, pos + 0.02,
    )}


def test_only_settled_off_on_the_spindle_succeeds():
    task = load(TASK)
    assert task.names == ("multimeter_body", "dial")
    assert task.free_objects == ()  # a spindle must not be jittered away from its casing
    assert task.success(states(task, 115))
    assert task.success(states(task, 114.54))
    for obj in (states(task, 0), states(task, -115), states(task, 110),
                states(task, 115, spin=0.2), states(task, 115, displacement=(0.02, 0, 0))):
        assert not task.success(obj)
    assert task.score(states(task, 0)) == pytest.approx(0)
    assert task.score(states(task, -90)) == 0
    assert 0 < task.score(states(task, 57.5)) < 1
    assert task.score(states(task, 115)) == pytest.approx(0.95)
    assert task.score(states(task, 115, spin=0.2)) < 1
    assert task.score(states(task, 115, displacement=(0.02, 0, 0))) == 0
    assert task.distance(states(task, 0)) == pytest.approx(math.radians(115))
    assert task.distance(states(task, 115)) < 1e-6


def test_demo_turns_only_counter_clockwise_and_is_reachable():
    task = load(TASK)
    plan = task.demo()
    assert set(plan) == {"right"}
    yaws = np.array([point.yaw for point in plan["right"]])
    assert np.all(np.diff(yaws) >= 0)
    assert yaws[-1] == pytest.approx(math.radians(115))
    arm = ArmPlan(plan["right"], station.BASES["right"])
    for step in range(math.ceil(arm.duration * 20) + 1):
        arm.action7(step / 20)
    assert arm.worst_ik_mm < 0.5
    assert arm.worst_ik_deg < 0.1


def write_task(tmp_path, body):
    (tmp_path / "asset.usda").write_text('#usda 1.0\n')
    path = tmp_path / "local_task.py"
    path.write_text(f"from {ADDON_NAME} import LocalUsdAsset as UsdAsset, Hinge\n" + body)
    return path


def test_local_paths_and_hinge_jitter_rules(tmp_path):
    path = write_task(tmp_path, f"OBJECTS = [UsdAsset('dial', pos=(0, 0, 1), usd_path={str(tmp_path / 'asset.usda')!r}, "
                      "mass=0.01, hinge=Hinge((0, 0, 0)))]")
    task = load(path)
    assert task.objects[0].usd_path == str(tmp_path / "asset.usda")
    assert task.free_objects == ()
    path = write_task(tmp_path, f"OBJECTS = [UsdAsset('loose', pos=(0, 0, 1), usd_path={str(tmp_path / 'asset.usda')!r}, mass=0.01)]")
    assert load(path).free_objects == ("loose",)


@pytest.mark.parametrize("options,message", [
    ("hinge=Hinge((0,0,0))", "requires.*mass"),
    ("mass=.01, hinge=Hinge((0,0,0), upper_degrees=200)", "hinge limits"),
    ("mass=.01, hinge=Hinge((0,0,0), lower_degrees=-10)", "counter-clockwise"),
    ("scale=(0,1,1)", "scale must be positive"),
    ("yaw=float('nan')", "yaw must be finite"),
])
def test_invalid_local_usd_specs_fail_before_starting_sim(tmp_path, options, message):
    path = write_task(tmp_path, f"OBJECTS = [UsdAsset('dial', pos=(0,0,1), usd_path={str(tmp_path / 'asset.usda')!r}, {options})]")
    with pytest.raises(ValueError, match=message):
        load(path)


def test_pinch_measurements_distinguish_a_handle_grasp_from_pushing():
    import copy
    module = load(TASK).module
    rows = [{
        'time': i * 0.05, 'gap': 0.008, 'dial_degrees': 0.0,
        'tip_left': {'force_local': [0, 5, 3], 'point_local': [0, -0.004, 0.002]},
        'tip_right': {'force_local': [0, -5, 3], 'point_local': [0, 0.004, 0.002]},
    } for i in range(20)]
    assert module.summarize_pinch(rows)['pinch_verified']
    one_finger = copy.deepcopy(rows)
    for row in one_finger:
        row['tip_right']['force_local'] = [0, 0, 0]
    assert not module.summarize_pinch(one_finger)['pinch_verified']
    disk_contact = copy.deepcopy(rows)
    for row in disk_contact:
        row['tip_left']['point_local'][2] = 0.0005
        row['tip_right']['point_local'][2] = 0.0005
    assert not module.summarize_pinch(disk_contact)['pinch_verified']
    same_side = copy.deepcopy(rows)
    for row in same_side:
        row['tip_right']['point_local'][1] = -0.004
    assert not module.summarize_pinch(same_side)['pinch_verified']
    top_press = copy.deepcopy(rows)
    for row in top_press:
        row['tip_left']['force_local'] = [0, 0.1, 5]
        row['tip_right']['force_local'] = [0, -0.1, 5]
    assert not module.summarize_pinch(top_press)['pinch_verified']



def test_full_credit_requires_native_success_termination():
    from types import SimpleNamespace
    from yam_arena.scoring import task_score

    scorer = task_score()
    for done, expected in [(False, 0.95), (True, 1.0)]:
        result = SimpleNamespace(terminated=done, termination_reason="success" if done else None,
                                 info={"score": 0.95})
        record = SimpleNamespace(steps=[SimpleNamespace(result=result)])
        assert scorer(record, None).value == expected
