"""A weekly (7-day) snap-lid pill organizer, extra-large size, lying loose on the table with its lids toward
the robot, and three caplets lying on the table in front of it.

The organizer is one light polypropylene strip of seven compartments in a row. Each day has its own lid,
hinged along the back edge. A short tongue hangs from the lid's front edge over the front wall, with a
fingernail ledge at its foot; a hook on the inside of the tongue snaps under a catch on the front wall. To
open a day you lift the ledge until the hook rides over the catch, swing the lid up past upright and let it
fall back onto its stop; to shut it you swing it down and press its front edge until the hook clicks under
the catch. All seven lids start snapped shut and every compartment is empty. Zero object jitter; 20 Hz.
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from yam_parts import Part, box, cyl
from yam_joints import Hinge, Link, Mechanism, Slide
from yam_arena import demo as _motion
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP

SCENE = "station"

# Release-and-stillness block, verbatim from hang_holder_on_keyhole_studs.py (see release-stillness.md).
# The task module must define: import math; import numpy as np; from yam_arena.constants import FINGER_OPEN;
# and SCENE ("station" or "tabletop"). Paste after the imports.

# Release and stillness
# Released means every gripper TCP is at least RELEASE_CLEAR from the object's box.
# Still means below the task's linear and angular speed limits for STILL_S seconds.
# Evaluate every sticky condition and timer at the top of score() on every call,
# before any early return, so interrupted conditions cannot retain elapsed time.
RELEASE_CLEAR = 0.030
STILL_S = 1.0
TEST_GRIPPERS = None


def _run_context():
    context = {"env": None, "t": None, "step": 0, "episode": 0}
    try:
        from yam_arena import inspect_embodiment as _ie
    except ImportError:
        return context
    cls = _ie.YamArenaEmbodiment
    if hasattr(cls, "_rc_run"):
        return cls._rc_run
    original_step, original_reset = cls._task_step, cls.reset

    def task_step(self, *args, **kwargs):
        context["env"], context["t"] = self._env, self._t
        context["step"] += 1
        return original_step(self, *args, **kwargs)

    def reset(self, *args, **kwargs):
        result = original_reset(self, *args, **kwargs)
        context["episode"] += 1
        context["t"] = 0.0
        context["env"] = self._env
        return result

    task_step._rc_wrapped = reset._rc_wrapped = True
    cls._rc_run = context
    cls._task_step, cls.reset = task_step, reset
    return context


_RUN = _run_context()
_MEMORY_EPISODE = None
_TIMERS = {}
_EVER = {}


def grippers():
    if TEST_GRIPPERS is not None:
        return TEST_GRIPPERS
    env = _RUN["env"]
    if env is None:
        return None
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, LEFT_BASE_POS, RIGHT_BASE_POS

    bases = station.BASES if SCENE == "station" else {"left": LEFT_BASE_POS, "right": RIGHT_BASE_POS}
    out = {}
    for arm in ("left", "right"):
        art = env.unwrapped.scene[arm + "_robot"]
        arm_ids = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
        finger_ids = art.find_joints(FINGER_JOINTS, preserve_order=True)[0]
        joints = art.data.joint_pos
        q = getattr(joints, "torch", joints)[0]
        if hasattr(q, "detach"):
            q = q.detach().cpu().numpy()
        q = np.asarray(q)
        tcp = ik.tcp_pose(q[arm_ids], bases[arm])[0]
        opening = float(np.clip(np.mean(q[finger_ids]) / FINGER_OPEN, 0.0, 1.0))
        out[arm] = (tcp, opening)
    return out


def released(state, clear=RELEASE_CLEAR):
    hands = grippers()
    if hands is None:
        return True  # Offline arm poses are unknown, so release is assumed.
    low, high = np.asarray(state.low), np.asarray(state.high)
    return all(np.linalg.norm(np.maximum(np.maximum(low - tcp, tcp - high), 0.0)) >= clear
               for tcp, _ in hands.values())


def _episode_memory():
    global _MEMORY_EPISODE
    if _MEMORY_EPISODE != _RUN["episode"]:
        _TIMERS.clear()
        _EVER.clear()
        _MEMORY_EPISODE = _RUN["episode"]


def held_for(key, cond):
    """Elapsed time over consecutive observed control steps; repeat calls are harmless."""
    _episode_memory()
    t, step = _RUN["t"], _RUN["step"]
    if t is None:
        return math.inf if cond else 0.0
    previous_step, start = _TIMERS.get(key, (None, None))
    if not cond:
        _TIMERS[key] = (step, None)
        return 0.0
    if start is None or step not in (previous_step, previous_step + 1) or t < start:
        start = t
    _TIMERS[key] = (step, start)
    return float(t - start)


def ever(key, cond):
    """Remember a satisfied condition until the next episode; offline each call stands alone."""
    if _RUN["t"] is None: return bool(cond)
    _episode_memory()
    _EVER[key] = bool(_EVER.get(key, False) or cond)
    return _EVER[key]



# ------------------------------------------------------------------ layout (metres)
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"

# Organizer frame: origin at the centre of its footprint on the table, x toward the back (away from the
# robot), y along the strip (Sunday at +y, the robot's left), z up. Listed size 7.4 x 1.5 x 0.9 in.
ORG_L, ORG_D = 0.188, 0.038           # length along y, depth front to back
PITCH = ORG_L / 7                     # 26.9 mm per day
WALL, FLOOR, DIV = 0.0012, 0.0012, 0.0010
RIM_Z = 0.0215                        # top of the walls; lids 1.5 mm thick on top: 23 mm overall
LID_T = 0.0015
FRONT_X = -ORG_D / 2                  # outer face of the front wall
INNER_X = ORG_D / 2 - WALL            # inner half-depth of a compartment (17.8 mm)
INNER_HALF_W = (PITCH - DIV) / 2      # inner half-width of a compartment (12.9 mm)
ORG_X, ORG_Y = 0.190, -0.0134         # Monday and Thursday sit either side of y = 0
ORG_POS = (ORG_X, ORG_Y, TOP)
DAYS = ("sun", "mon", "tue", "wed", "thu", "fri", "sat")
DAY_NAMES = ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")
LETTERS = "SMTWTFS"
MON, THU = 1, 4


def day_y(k):
    """Organizer-frame y of day k's compartment centre (0 = Sunday, at +y)."""
    return (3 - k) * PITCH


# Lid, tongue and catch (organizer frame, lid shut). The hinge runs along the back top edge.
HINGE_X, HINGE_Z = ORG_D / 2, RIM_Z
LID_OPEN_DEG = 110.0                  # an open lid falls back past upright onto this stop
LID_FRONT_X = FRONT_X - 0.0010        # the lid plate overhangs the front wall by 1 mm
TONGUE_IN = FRONT_X - 0.0018          # tongue: 1.2 mm thick, 12 mm wide, hanging 9 mm below the lid
TONGUE_OUT = TONGUE_IN - 0.0012
TONGUE_Z0 = 0.0125
TONGUE_W = 0.012
LEDGE_OUT = TONGUE_OUT - 0.0025       # fingernail ledge: 2.5 mm proud of the tongue at its foot
LEDGE_Z = (TONGUE_Z0, TONGUE_Z0 + 0.0012)
CATCH_R, HOOK_R = 0.0015, 0.0012      # catch: a rounded ridge on the front wall; hook: a rounded rib on the tongue
CATCH_Z = 0.0165
CATCH_X = FRONT_X
HOOK_X = TONGUE_IN + 0.0016 - HOOK_R  # the hook stands 1.6 mm in from the tongue
_dx = CATCH_X - HOOK_X
HOOK_Z = CATCH_Z - math.sqrt((CATCH_R + HOOK_R) ** 2 - _dx ** 2) - 0.0003   # 0.3 mm play under the catch
CATCH_K, CATCH_PRELOAD = 1200.0, 0.2  # N/m, N: the sprung catch (the flex of the real snap, see fidelity)
CATCH_TRAVEL = 0.0025
LID_MASS, CATCH_MASS, BASE_MASS = 0.0020, 0.0005, 0.0175      # 35 g in all

BLUE = (0.42, 0.62, 0.90)
LID_BLUE = (0.55, 0.74, 0.96)
WHITE = (0.97, 0.97, 0.97)
CAPLET_WHITE = (0.93, 0.92, 0.86)


def _letter(ch, y0):
    """Raised capital on a lid, read from the robot's side: up is +x, right is -y. 14 x 10 mm, 2 mm strokes."""
    h, w, s, z = 0.014, 0.010, 0.002, RIM_Z + LID_T + 0.0002
    x0 = -0.004                                    # letter centre, a little ahead of the lid centre
    def bar(u0, u1, v0, v1):                       # u: rightward (-y) from the left edge, v: up (+x) from the bottom
        return box((x0 - h / 2 + (v0 + v1) / 2, y0 + w / 2 - (u0 + u1) / 2, z), (v1 - v0, u1 - u0, 0.0004), WHITE)
    full_h, half = h, h / 2
    if ch == "S":
        return [bar(0, w, full_h - s, full_h), bar(0, s, half, full_h), bar(0, w, half - s / 2, half + s / 2),
                bar(w - s, w, 0, half), bar(0, w, 0, s)]
    if ch == "M":
        return [bar(0, s, 0, full_h), bar(w - s, w, 0, full_h), bar(0, w, full_h - s, full_h),
                bar(w / 2 - s / 2, w / 2 + s / 2, half, full_h)]
    if ch == "W":
        return [bar(0, s, 0, full_h), bar(w - s, w, 0, full_h), bar(0, w, 0, s),
                bar(w / 2 - s / 2, w / 2 + s / 2, 0, half)]
    if ch == "T":
        return [bar(0, w, full_h - s, full_h), bar(w / 2 - s / 2, w / 2 + s / 2, 0, full_h)]
    if ch == "F":
        return [bar(0, s, 0, full_h), bar(0, w, full_h - s, full_h), bar(0, w * 0.75, half - s / 2, half + s / 2)]
    raise ValueError(ch)


def _lid_shapes(k):
    y = day_y(k)
    w = PITCH - 0.0010
    return [
        box(((LID_FRONT_X + HINGE_X) / 2, y, RIM_Z + LID_T / 2), (HINGE_X - LID_FRONT_X, w, LID_T)),     # lid plate
        box(((TONGUE_IN + TONGUE_OUT) / 2, y, (TONGUE_Z0 + RIM_Z) / 2), (TONGUE_IN - TONGUE_OUT, TONGUE_W,
                                                                         RIM_Z - TONGUE_Z0)),         # tongue
        box(((TONGUE_OUT + LEDGE_OUT) / 2, y, sum(LEDGE_Z) / 2), (TONGUE_OUT - LEDGE_OUT, TONGUE_W,
                                                                LEDGE_Z[1] - LEDGE_Z[0])),          # ledge
        cyl((HOOK_X, y, HOOK_Z), HOOK_R, 0.008, "y"),                                                # hook rib
    ] + _letter(LETTERS[k], y)


def _base_shapes():
    shapes = [
        box((0.0, 0.0, FLOOR / 2), (ORG_D, ORG_L, FLOOR)),
        box((FRONT_X + WALL / 2, 0.0, (FLOOR + RIM_Z) / 2), (WALL, ORG_L, RIM_Z - FLOOR)),
        box((ORG_D / 2 - WALL / 2, 0.0, (FLOOR + RIM_Z) / 2), (WALL, ORG_L, RIM_Z - FLOOR)),
        box((0.0, ORG_L / 2 - WALL / 2, (FLOOR + RIM_Z) / 2), (ORG_D - 2 * WALL, WALL, RIM_Z - FLOOR)),
        box((0.0, -ORG_L / 2 + WALL / 2, (FLOOR + RIM_Z) / 2), (ORG_D - 2 * WALL, WALL, RIM_Z - FLOOR)),
    ]
    for k in range(6):                     # dividers between the days
        shapes.append(box((0.0, day_y(k) - PITCH / 2, (FLOOR + RIM_Z) / 2), (ORG_D - 2 * WALL, DIV, RIM_Z - FLOOR)))
    return shapes


_links = []
for _k, _day in enumerate(DAYS):
    _links.append(Link(f"lid_{_day}", _lid_shapes(_k),
                       Hinge("y", (HINGE_X, day_y(_k), HINGE_Z), 0.0, LID_OPEN_DEG, damping=1e-4, armature=1e-6),
                       mass=LID_MASS, color=LID_BLUE, friction=(0.6, 0.5)))
    _links.append(Link(f"catch_{_day}", [cyl((CATCH_X, day_y(_k), CATCH_Z), CATCH_R, 0.008, "y")],
                       Slide("x", (CATCH_X, day_y(_k), CATCH_Z), 0.0, CATCH_TRAVEL, damping=0.3, spring=CATCH_K,
                             rest=-CATCH_PRELOAD / CATCH_K, armature=0.002),
                       mass=CATCH_MASS, color=BLUE, friction=(0.15, 0.12)))

ORGANIZER = Mechanism("organizer", ORG_POS, _base_shapes(), _links, mass=BASE_MASS, color=BLUE,
                      friction=(0.4, 0.35), contact_offset=0.001, self_collide=True)

# Caplets: white film-coated multivitamin caplets, 19 x 9 x 7 mm (a 9 mm-wide stadium), 1.3 g each, lying
# flat on the table with their long axis pointing away from the robot.
CAPLET_L, CAPLET_W, CAPLET_H, CAPLET_MASS = 0.019, 0.009, 0.007, 0.0013
CAPLET_START = {"caplet_a": (0.085, 0.045), "caplet_b": (0.085, -0.045), "caplet_c": (0.085, 0.0)}


def _caplet(name):
    x, y = CAPLET_START[name]
    core = CAPLET_L - CAPLET_W
    return Part(name, (x, y, TOP + CAPLET_H / 2),
                [box((0, 0, 0), (core, CAPLET_W, CAPLET_H)),
                 cyl((core / 2, 0, 0), CAPLET_W / 2, CAPLET_H, "z"), cyl((-core / 2, 0, 0), CAPLET_W / 2, CAPLET_H, "z")],
                mass=CAPLET_MASS, color=CAPLET_WHITE, friction=(0.4, 0.35), contact_offset=0.001, grip=0)


CAPLETS = tuple(_caplet(n) for n in CAPLET_START)
CAPLET_NAMES = tuple(CAPLET_START)
PRESSED_PARTS = ("organizer",)
OBJECTS = [ORGANIZER, *CAPLETS]


def org_point(p_local):
    """World point of an organizer-frame point at the start pose."""
    return np.array([ORG_X + p_local[0], ORG_Y + p_local[1], TOP + p_local[2]])


def about_hinge(p_local, deg):
    """An organizer-frame point on a lid (given with the lid shut) after the lid has opened by deg."""
    a = math.radians(deg)
    dx, dz = p_local[0] - HINGE_X, p_local[2] - HINGE_Z
    return np.array([HINGE_X + dx * math.cos(a) + dz * math.sin(a), p_local[1], HINGE_Z - dx * math.sin(a) + dz * math.cos(a)])


# Reference layouts for the scorer tests (states.yaml): a caplet lying on a compartment floor.
def caplet_in(k):
    return tuple(float(v) for v in org_point((0.0, day_y(k), FLOOR + CAPLET_H / 2)))


IN_MON, IN_THU, IN_TUE, IN_SUN = caplet_in(MON), caplet_in(THU), caplet_in(2), caplet_in(0)
ON_MON_LID = tuple(float(v) for v in org_point((0.0, day_y(MON), RIM_Z + LID_T + 0.0006 + CAPLET_H / 2)))

# ------------------------------------------------------------------ scoring
SNAP_MARGIN = 0.0005              # snapped: the hook's centre at least 0.5 mm below the catch's centre
FLAT_DEG, MARK_M = 3.0, 0.030     # organizer flat on the table and within 30 mm of where it lay
TURNED_DEG = 45.0                 # sticky zero: organizer tipped over
DROP_Z = TOP - 0.020              # sticky zero: anything below the table top
STEP_M, STEP_DEG = 0.0005, 0.25
TEST_STICKY = ()                  # offline tests: sticky keys that already fired earlier in the episode


def sticky(key, cond):
    return ever(key, bool(cond) or key in TEST_STICKY)


def lid_snapped(org, k):
    """The lid's hook is under its catch: the lid cannot lift without pushing the catch back."""
    day = DAYS[k]
    hook = org.to_local(org.link_to_world(f"lid_{day}", (HOOK_X, day_y(k), HOOK_Z)))
    catch = org.to_local(org.link_to_world(f"catch_{day}", (CATCH_X, day_y(k), CATCH_Z)))
    return bool(hook[2] <= catch[2] - SNAP_MARGIN and hook[0] >= catch[0] - (CATCH_R + HOOK_R)
                and abs(hook[0] - catch[0]) <= CATCH_R + HOOK_R)


def compartment_of(org, caplet):
    """Index of the compartment whose inside holds the caplet's centre, or None."""
    p = org.to_local(caplet.origin)
    if not (abs(p[0]) <= INNER_X and FLOOR <= p[2] <= RIM_Z):
        return None
    for k in range(7):
        if abs(p[1] - day_y(k)) <= INNER_HALF_W:
            return k
    return None


def counts(objects):
    org = objects["organizer"]
    n = [0] * 7
    for name in CAPLET_NAMES:
        k = compartment_of(org, objects[name])
        if k is not None:
            n[k] += 1
    return n


def organizer_ok(org):
    d = np.asarray(org.origin, float) - np.asarray(ORG_POS, float)
    return bool(org.tilt_deg() <= FLAT_DEG and abs(d[2]) <= 0.003 and np.linalg.norm(d[:2]) <= MARK_M)


_POSE_EPISODE = None
_POSE_TOKEN = None
_PREVIOUS = None
_QUIET = False


def at_rest(objects):
    """Pose differences between consecutive control steps (contact chatter makes joint speeds unreliable)."""
    global _POSE_EPISODE, _POSE_TOKEN, _PREVIOUS, _QUIET
    org = objects["organizer"]
    caps = [objects[n] for n in CAPLET_NAMES]
    if _RUN["t"] is None:
        return org.resting() and all(c.resting() for c in caps)
    if _POSE_EPISODE != _RUN["episode"]:
        _PREVIOUS, _POSE_TOKEN, _QUIET = None, None, False
        _POSE_EPISODE = _RUN["episode"]
    token = (_RUN["step"], float(_RUN["t"]))
    if token != _POSE_TOKEN:
        pose = (np.r_[org.to_world((0, 0, 0)), org.to_world((0, 0.09, 0)), *[c.origin for c in caps]],
                np.array([org.angle_deg(f"lid_{d}") for d in DAYS]))
        _QUIET = False
        if _PREVIOUS is not None:
            step, old = _PREVIOUS
            _QUIET = bool(token[0] == step + 1 and np.all(np.abs(pose[0] - old[0]) < STEP_M)
                          and np.all(np.abs(pose[1] - old[1]) < STEP_DEG))
        _PREVIOUS = (token[0], pose)
        _POSE_TOKEN = token
    return _QUIET


def status(objects):
    org = objects["organizer"]
    n = counts(objects)
    snapped = [lid_snapped(org, k) for k in range(7)]
    return dict(counts=n, snapped=snapped, flat=organizer_ok(org))


def score(objects):
    org = objects["organizer"]
    caps = [objects[n] for n in CAPLET_NAMES]
    st = status(objects)
    n = st["counts"]
    wrong_day = sticky("a caplet in another day", any(n[k] for k in range(7) if k not in (MON, THU)))
    doubled = sticky("two caplets in one day", max(n) >= 2)
    turned = sticky("organizer turned over", org.tilt_deg() > TURNED_DEG)
    fell = sticky("off the table", org.origin[2] < DROP_Z or any(c.origin[2] < DROP_Z for c in caps))
    quiet = at_rest(objects)
    done = released(org) and quiet
    seconds = held_for("released and still", done)
    if wrong_day or doubled or turned or fell:
        return 0.0
    if not (done and seconds + 1e-9 >= STILL_S):
        return 0.0
    if n[MON] == 1 and n[THU] == 1:
        if not st["flat"]:
            return 0.5
        unsnapped = 7 - sum(st["snapped"])
        return 1.0 if unsnapped == 0 else 0.75 if unsnapped == 1 else 0.5
    one = [k for k in (MON, THU) if n[k] == 1]
    if len(one) == 1 and st["snapped"][one[0]]:
        return 0.25
    return 0.0


def success(objects):
    return score(objects) >= 1.0


# ------------------------------------------------------------------ scripted demo (open loop: start layout and time)
# Tilted-wrist waypoints: a float yaw (for the offline checker's coarse replay) carrying the full grip rotation.
class GripOrientation(float):
    def __new__(cls, matrix):
        obj = super().__new__(cls, math.atan2(-matrix[0, 0], matrix[1, 0]))
        obj.matrix = np.asarray(matrix, float).copy()
        return obj


_original_grip_rotation = getattr(_motion.grip_rotation, "_organizer_original", _motion.grip_rotation)


def _grip_rotation(yaw=0.0):
    matrix = getattr(yaw, "matrix", None)
    if isinstance(matrix, np.ndarray) and matrix.shape == (3, 3):
        return matrix
    return _original_grip_rotation(yaw)


_grip_rotation._organizer_original = _original_grip_rotation
_motion.grip_rotation = _grip_rotation

YAW = {"left": -math.pi / 2, "right": math.pi / 2}      # both hands close their pads along world x
# Tilt of the closed fingertips from vertical (tips forward), by lid angle. More than about 55 deg puts the
# gripper body on the table; the fingertip's upper face then slopes 48 deg, so the edge gets 1.7 mm under the ledge.
NAIL_TILT = {0: 52.0, 9: 52.0, 20: 50.0, 35: 47.0, 50: 44.0, 65: 40.0, 80: 35.0, 95: 35.0, 104: 35.0}
NAIL_X = TONGUE_OUT - 0.0003      # fingertip edge in under the ledge, against the tongue
NAIL_Z = 0.0098                   # and 2.7 mm below the ledge's underside (12.5 mm up)
TIP = 0.013                       # pad tips past the TCP along the tool axis
HALF_T = 0.0035                   # half the thickness of the closed fingertips
# The holding hand presses two spread fingertips down on the lids at the far end of the strip, 40 mm apart
# along it (one 3 mm from the end), so the strip cannot pivot up about the hand while the other lifts a ledge.
HOLD_SPREAD = 0.040
HOLD_FINGER = -HOLD_SPREAD / 2
HOLD_Y = ORG_L / 2 - 0.003 - HOLD_SPREAD / 2              # organizer-frame |y| of the holding hand's TCP
HOLD_PRESS = 0.012                # fingertips commanded 12 mm below the lid tops (about 8 N)
CAPLET_FINGER, CAPLET_OPEN, CAPLET_LETGO = -0.0078, -0.016, -0.011
# Opening Monday nudges the strip about 0.7 mm away from the robot (and 0.4 mm left), measured on the GPU; the
# open-loop Thursday fingertip aims where the strip then lies.
DRIFT = {4: np.array([0.0007, 0.0004, 0.0])}
NAIL_PUSH = 0.006                 # the fingertip is commanded this far into the ledge to pop the snap
DEMO_PACE = 1.0


def _ry(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def nail_rot(arm, deg=0):
    """Closed fingers turned so the tips point forward (+x) and down (65 deg from vertical under the ledge,
    standing up to 35 deg as the lid swings open)."""
    return GripOrientation(_ry(-NAIL_TILT[deg]) @ _original_grip_rotation(YAW[arm]))


def nail_tcp(k, deg, push=0.0, back=0.0, down=0.0):
    """TCP that puts the fingertips' front-top edge just under day k's ledge with the lid open deg."""
    a = math.radians(NAIL_TILT[deg])
    tool = np.array([math.sin(a), 0.0, -math.cos(a)])          # tip direction
    face = np.array([math.cos(a), 0.0, math.sin(a)])           # front-top face normal
    corner = np.array([NAIL_X, day_y(k), NAIL_Z])
    p = about_hinge(corner, deg)
    n = about_hinge(corner + np.array([0, 0, 1.0]), deg) - about_hinge(corner, deg)
    p = p + push * n + np.array([-back, 0.0, -down])
    return org_point(p) + DRIFT.get(k, 0.0) - TIP * tool - HALF_T * face


def demo():
    plan = {"left": [], "right": []}
    st = {}

    def add(dt, label, left=None, right=None, lf=None, rf=None, lr=None, rr=None):
        for arm, pos, f, r in (("left", left, lf, lr), ("right", right, rf, rr)):
            if pos is not None:
                st[arm] = np.asarray(pos, float)
            if f is not None:
                st[arm + "f"] = f
            if r is not None:
                st[arm + "r"] = r
            plan[arm].append(Waypoint(dt * DEMO_PACE, tuple(float(v) for v in st[arm]), st[arm + "f"], label,
                                      st[arm + "r"]))

    up = np.array([0.0, 0.0, 1.0])
    down_l, down_r = YAW["left"], YAW["right"]
    park = {"left": np.array([0.12, 0.20, TOP + 0.12]), "right": np.array([0.12, -0.20, TOP + 0.12])}
    st.update(left=park["left"], leftf=FINGER_OPEN, leftr=down_l, right=park["right"], rightf=FINGER_OPEN,
              rightr=down_r)

    def hold_pose(k):
        side = 1.0 if day_y(k) > 0 else -1.0
        return org_point((0.0, side * HOLD_Y, RIM_Z + LID_T + TIP - HOLD_PRESS))

    def open_lid(arm, holder, k, hold_k):
        nail = nail_rot(arm)
        dn = 0.0                                   # holding hand: pads spread along the strip
        kw = {"left": "left", "right": "right"}
        hp = hold_pose(hold_k)
        add(2.0, f"{holder}: over the {DAY_NAMES[hold_k]} end; {arm}: fingertips in front of the {DAY_NAMES[k]} ledge",
            **{kw[holder]: hp + 0.05 * up, kw[arm]: nail_tcp(k, 0, back=0.010, down=0.0015) + 0.03 * up},
            **{holder[0] + "f": HOLD_FINGER, arm[0] + "f": FINGER_CLOSED, holder[0] + "r": dn, arm[0] + "r": nail})
        add(1.2, f"{holder}: press two fingertips on the strip's end; {arm}: down in front of the ledge",
            **{kw[holder]: hp, kw[arm]: nail_tcp(k, 0, back=0.010, down=0.0015)})
        add(0.6, f"{holder}: hold the strip down")
        add(0.8, f"{arm}: fingertips under the {DAY_NAMES[k]} ledge", **{kw[arm]: nail_tcp(k, 0)})
        add(1.2, "lift the ledge until the hook clears the catch",
            **{kw[arm]: nail_tcp(k, 9, push=NAIL_PUSH) + np.array([0.006, 0.0, 0.0]), arm[0] + "r": nail_rot(arm, 9)})
        for deg in (20, 35):
            add(0.4, f"swing the lid open {deg}", **{kw[arm]: nail_tcp(k, deg, push=0.002), arm[0] + "r": nail_rot(arm, deg)})
        side = 1.0 if hp[1] > ORG_Y else -1.0
        for deg in (50, 65, 80, 95, 104):
            extra = {kw[holder]: hp + np.array([0.0, side * 0.04, 0.05])} if deg == 50 else {}
            add(0.4, f"swing the lid open {deg}" + (f"; {holder}: lets go and moves aside" if extra else ""),
                **{kw[arm]: nail_tcp(k, deg, push=0.002), arm[0] + "r": nail_rot(arm, deg)}, **extra)
        add(0.6, f"{arm}: fingertips out from under the lid",
            **{kw[arm]: nail_tcp(k, 104) + np.array([-0.025, 0.0, -0.010])})

    def fill(arm, k, name):
        dn = YAW[arm]
        cx, cy = CAPLET_START[name]
        grab = np.array([cx, cy, TOP + CAPLET_H / 2 + TIP - 0.0015])
        add(1.6, f"{arm}: over a caplet", **{arm: grab + 0.05 * up, arm[0] + "f": CAPLET_OPEN, arm[0] + "r": dn})
        add(1.0, f"{arm}: down around the caplet", **{arm: grab})
        add(0.6, f"{arm}: pinch the caplet end to end", **{arm[0] + "f": CAPLET_FINGER})
        add(0.9, f"{arm}: lift", **{arm: grab + 0.06 * up})
        drop = org_point((0.0, day_y(k), RIM_Z))              # pad tips 13 mm below the rim
        add(1.6, f"{arm}: over the open {DAY_NAMES[k]} compartment", **{arm: drop + 0.035 * up})
        add(1.0, f"{arm}: down into it", **{arm: drop})
        add(0.5, f"{arm}: let go of the caplet", **{arm[0] + "f": CAPLET_LETGO})
        add(0.8, f"{arm}: up, fingertips clear above the open lid", **{arm: drop + 0.075 * up})

    def close_lid(arm, k):
        dn = YAW[arm]
        behind = org_point((0.046, day_y(k), 0.058))          # 6 mm behind the open lid's top end
        add(1.2, f"{arm}: closed fingers behind the open {DAY_NAMES[k]} lid",
            **{arm: behind + 0.03 * up, arm[0] + "f": FINGER_CLOSED, arm[0] + "r": dn})
        add(0.8, f"{arm}: down behind the lid", **{arm: behind})
        add(1.2, f"{arm}: push the lid forward past upright", **{arm: org_point((0.014, day_y(k), 0.059))})
        add(0.6, f"{arm}: up out of its way; the lid falls shut onto the catch",
            **{arm: org_point((0.020, day_y(k), 0.100))})
        add(1.2, "lid settles")
        press = org_point((-0.010, day_y(k), RIM_Z + LID_T + TIP))
        add(1.0, f"{arm}: over the lid's front edge", **{arm: press + 0.02 * up})
        add(1.0, f"{arm}: press the lid down until it clicks", **{arm: press - 0.007 * up})
        add(0.5, f"{arm}: hold")
        add(0.8, f"{arm}: up", **{arm: press + 0.06 * up})

    # Monday: the right hand holds the Saturday end down while the left opens Monday.
    add(2.5, "approach")
    open_lid("left", "right", MON, 6)
    add(1.2, "right parks", right=park["right"])
    fill("left", MON, "caplet_a")
    close_lid("left", MON)
    # Thursday: the left hand holds the Sunday end down while the right opens Thursday.
    open_lid("right", "left", THU, 0)
    add(1.2, "left parks", left=park["left"])
    fill("right", THU, "caplet_b")
    close_lid("right", THU)
    add(1.2, "both hands back", left=park["left"], right=park["right"])
    add(3.5, "stillness tail")
    return plan


def demo_duration():
    return sum(w.duration for w in demo()["right"])
