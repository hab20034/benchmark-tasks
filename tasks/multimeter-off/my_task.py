"""Turn the digital multimeter's dial counter-clockwise until its small pointer dot aligns with OFF.

The original calibrated station has both YAM arms, the table, and all three cameras. The
NVIDIA testingTool multimeter is its only task asset, rotated 180 degrees about Z. The
meter casing is fixed to the table; the dial is a passive Z hinge with a one-way stop.
Use the right gripper to pinch the dial, turn it, release it, and retreat. Do not lift the
meter. OFF is the 115-degree counter-clockwise stop from the initial dial position.

Both policies use this file and the same success/score functions:
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=/path/to/multimeter_off_submission/my_task.py ...
inspect-robots run --embodiment yam_arena --policy dream -E task=/path/to/multimeter_off_submission/my_task.py ...

All custom asset and hinge code lives in this submission folder. multimeter_addon.py
loads the simulation adapter after Isaac Sim starts; no installed environment files are
edited. No dial drive or grasp assistance is used.

Direct pinch verification: python my_task.py --window
"""

import hashlib
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP

# Load the add-on beside this task, independent of cwd or PYTHONPATH. The native task
# loader imports this file before Isaac Sim starts, so the add-on defers simulator imports.
HERE = Path(__file__).resolve().parent
_addon_path = HERE / "multimeter_addon.py"
_addon_name = "_multimeter_off_addon_" + hashlib.sha256(str(_addon_path).encode()).hexdigest()[:12]
_addon = sys.modules.get(_addon_name)
if _addon is None:
    _spec = importlib.util.spec_from_file_location(_addon_name, _addon_path)
    _addon = importlib.util.module_from_spec(_spec)
    sys.modules[_addon_name] = _addon
    _spec.loader.exec_module(_addon)
_addon.install()
Hinge, LocalUsdAsset = _addon.Hinge, _addon.LocalUsdAsset

SCENE = "station"
SUCCESS_HOLD_S = 2.0
SCORE_OVER = "end"
OFF_DEGREES = 115.0
ANGLE_TOLERANCE_DEGREES = 3.0

ASSETS = HERE / "assets"

# Native bounds of the imported USD, in metres. Keep native size and centre its complete
# bounding box (including leads) at (0.30, 0.0) on the table, as in the original scene.
LOW = np.array([-0.09356080973539813, -0.2919746935061783, 0.000008086208520307822])
HIGH = np.array([0.09356076782550352, 0.04414743557572365, 0.04047299176454544])
YAW = math.pi
ROTATION = np.diag([-1.0, -1.0, 1.0])
ORIGIN = np.array([0.30, 0.0, TABLE_TOP[SCENE] + 0.001 - LOW[2]])
ORIGIN[:2] -= (ROTATION @ ((LOW + HIGH) / 2))[:2]
PIVOT_LOCAL = (-0.00899158977, 0.000009714819498, 0.03652296215)
PIVOT_WORLD = ORIGIN + ROTATION @ np.array(PIVOT_LOCAL)

MULTIMETER = LocalUsdAsset(
    "multimeter_body", pos=tuple(ORIGIN), yaw=YAW, usd_path=str(ASSETS / "multimeter_body.usd"),
)
DIAL = LocalUsdAsset(
    "dial", pos=tuple(ORIGIN), yaw=YAW, usd_path=str(ASSETS / "multimeter_dial.usd"), mass=0.004559,
    hinge=Hinge(pivot=PIVOT_LOCAL, upper_degrees=OFF_DEGREES),
    collision_filter=("multimeter_body",), friction=(2.0, 1.5),
)
OBJECTS = [MULTIMETER, DIAL]


def dial_degrees(objects):
    """Actual dial rotation from its reset pose, positive counter-clockwise from above."""
    x, y, z, w = objects["dial"].quat_xyzw
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return math.degrees(math.atan2(math.sin(yaw - YAW), math.cos(yaw - YAW)))


def _on_spindle(objects):
    dial = objects["dial"]
    x, y, _, _ = dial.quat_xyzw
    return bool(
        np.linalg.norm(dial.pos - PIVOT_WORLD) < 0.005
        and 1 - 2 * (x * x + y * y) > math.cos(math.radians(5))
    )


def success(objects):
    """The actual dial is within 3 degrees of OFF and still on its spindle, settled."""
    return bool(
        _on_spindle(objects)
        and abs(dial_degrees(objects) - OFF_DEGREES) <= ANGLE_TOLERANCE_DEGREES
        and objects["dial"].resting(max_speed=0.005, max_spin=0.05)
    )


def distance(objects):
    """Angular error to OFF, in radians (this rotational task has no translational goal)."""
    return math.radians(abs(OFF_DEGREES - dial_degrees(objects)))


def score(objects):
    """Up to 0.95 for progress; the native success scorer awards 1 after the two-second hold."""
    if not _on_spindle(objects):
        return 0.0
    # Full credit belongs to the native success scorer after SUCCESS_HOLD_S.
    # Returning 1 here would bypass the hold when a trial stops early.
    return float(0.95 * np.clip(dial_degrees(objects) / OFF_DEGREES, 0.0, 1.0))


def demo():
    """Right arm: approach, pinch, turn counter-clockwise in small yaw increments, release, retreat."""
    xy = tuple(PIVOT_WORLD[:2])
    grasp_z = float(PIVOT_WORLD[2] + 0.0143)
    hover_z = grasp_z + 0.075
    points = [
        Waypoint(2.5, (*xy, hover_z), FINGER_OPEN, "approach"),
        Waypoint(0.8, (*xy, hover_z), -0.008, "pre-shape around handle"),
        Waypoint(2.5, (*xy, grasp_z), -0.008, "descend around handle"),
        Waypoint(1.5, (*xy, grasp_z), FINGER_CLOSED, "grip dial"),
    ]
    for degrees in np.linspace(0, OFF_DEGREES, math.ceil(OFF_DEGREES / 5) + 1)[1:]:
        points.append(Waypoint(
            0.35, (*xy, grasp_z), FINGER_CLOSED, "turn counter-clockwise", math.radians(degrees),
        ))
    target = math.radians(OFF_DEGREES)
    points += [
        Waypoint(1.0, (*xy, grasp_z), FINGER_CLOSED, "hold OFF", target),
        # Release the narrow handle without sweeping the jaws into nearby buttons.
        # Hold a roughly 16 mm gap until the pad tips are 35 mm higher.
        Waypoint(0.6, (*xy, grasp_z), -0.008, "release handle narrowly", target),
        Waypoint(1.5, (*xy, grasp_z + 0.035), -0.008, "lift clear of buttons", target),
        Waypoint(0.6, (*xy, grasp_z + 0.035), FINGER_OPEN, "open above buttons", target),
        Waypoint(1.0, (*xy, hover_z), FINGER_OPEN, "retreat", target),
    ]
    return {"right": points}


def summarize_pinch(rows):
    """Require opposing lateral forces above the disk, not merely any contact with the dial."""
    dt = float(np.median([b["time"] - a["time"] for a, b in zip(rows, rows[1:])
                          if b["time"] > a["time"]])) if len(rows) > 1 else 0.05
    valid = []
    streak = before_streak = longest = longest_before = 0
    previous_time = None
    for row in rows:
        angle = math.radians(row["dial_degrees"]) + math.pi
        rotation = np.array([[math.cos(angle), -math.sin(angle), 0],
                             [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
        contacts = [row[name] for name in ("tip_left", "tip_right")]
        forces = [np.asarray(c["force_local"]) if "force_local" in c
                  else rotation.T @ np.asarray(c["force"]) for c in contacts]
        points = [np.asarray(c["point_local"]) for c in contacts]
        good = (
            all(np.isfinite(f).all() and np.linalg.norm(f) > 0.1
                and abs(f[1]) > 0.6 * np.linalg.norm(f) for f in forces)
            and forces[0][1] * forces[1][1] < 0
            and all(np.isfinite(p).all() and 0.001 < p[2] < 0.0035
                    and abs(p[1]) < 0.006 and abs(p[0]) < 0.023 for p in points)
            and points[0][1] * points[1][1] < 0
            and 0 < row["gap"] < 0.012
        )
        if previous_time is not None and not 0 < row["time"] - previous_time < 1.5 * dt:
            streak = before_streak = 0
        previous_time = row["time"]
        streak = streak + 1 if good else 0
        near_start = abs(math.degrees(math.atan2(math.sin(angle - math.pi), math.cos(angle - math.pi)))) < 5
        before_streak = before_streak + 1 if good and near_start else 0
        longest = max(longest, streak)
        longest_before = max(longest_before, before_streak)
        if good:
            valid.append(row)
    return {
        "pinch_verified": longest * dt >= 0.5 and longest_before * dt >= 0.25,
        "bilateral_handle_samples": len(valid), "total_samples": len(rows),
        "bilateral_handle_seconds": len(valid) * dt,
        "longest_bilateral_seconds": longest * dt,
        "longest_near_start_seconds": longest_before * dt,
        "median_pinched_gap_mm": float(np.median([r["gap"] for r in valid]) * 1000) if valid else None,
    }



def _run_pinch_check():
    """Optional direct execution: run this same task with contact verification."""
    import argparse
    import datetime
    import json
    import os
    import shutil
    import subprocess

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--out", type=Path, default=Path.home() / "yam-eval/multimeter-pinch-check")
    args = parser.parse_args()
    cli = shutil.which("inspect-robots")
    if cli is None:
        parser.error("Activate the existing challenge environment first (source ~/IsaacLab-Arena/yam.env)")
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S_%f")
    out = args.out.expanduser().resolve() / stamp
    out.mkdir(parents=True)
    contacts = out / "contacts.jsonl"
    env = dict(os.environ, MULTIMETER_PINCH_LOG=str(contacts))
    command = [
        cli, "run", "--embodiment", "yam_arena", "--policy", "yam_scripted",
        "-E", f"task={Path(__file__).resolve()}",
        "-E", "object_jitter=0", "-E", "control_hz=20",
        "-E", "cam_width=320", "-E", "cam_height=176",
        "--instruction", "Turn the multimeter dial counter-clockwise to OFF",
        "--scorer", "task_score", "--max-steps", "600", "--no-store-frames",
        "--log-dir", str(out / "eval"),
    ]
    if args.window:
        command += ["-E", "headless=false"]
    print(f"Running pinch check; output: {out}", flush=True)
    with (out / "simulator.log").open("w") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not contacts.exists():
        raise RuntimeError(f"No contact measurements; inspect {out / 'simulator.log'}")
    rows = [json.loads(line) for line in contacts.read_text().splitlines()]
    result = summarize_pinch(rows)
    logs = [p for p in (out / "eval").glob("*.json") if ".live." not in p.name]
    evaluation = json.loads(max(logs, key=lambda p: p.stat().st_mtime).read_text())
    result["task_score"] = evaluation["results"]["metrics"].get("task_score")
    result["evaluation_status"] = evaluation["status"]
    (out / "pinch_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["pinch_verified"] or result["task_score"] != 1.0 or evaluation["status"] != "success":
        raise RuntimeError("Pinch/turn verification failed; see measurements and close-up frames")



if __name__ == "__main__":
    _run_pinch_check()
