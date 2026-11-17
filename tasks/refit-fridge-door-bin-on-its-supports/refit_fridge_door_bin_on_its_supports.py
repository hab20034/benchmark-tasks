"""Refitting a refrigerator door bin: a real-size clear door bin (457 x 203 x 127 mm, the size of common 18 x 8 x 5 in
replacement bins), empty after cleaning, stands on a rubber mat in front of a real-size section of a fridge door
liner clamped upright on the repair bench, its inside facing the robot. The section keeps the two side walls (dikes)
and, on each dike, the moulded support boss of the lowest bin position. Each bin end wall has a pocket, open at the
bottom, that must drop over its boss; the last 4 mm of travel is a snug press fit that needs a push. If an end misses,
its end wall rests on top of the boss and the bin sits about 25 mm high there. Zero object jitter; 20 Hz.
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from yam_parts import Part, box
from yam_joints import Link, Slide
from yam_mesh import Mesh, MeshMechanism
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP

SCENE = "station"

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



# ------------------------------------------------------------------ layout (metres; world x forward, y left, z up)
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
MM = 0.001

# Door bin (bin frame = part frame: x from the front face toward the door, y along the bin, z up from the bottom).
BIN_L, BIN_D, BIN_H = 457.0 * MM, 203.0 * MM, 127.0 * MM   # 18 x 8 x 5 in
FRONT_H = 90.0 * MM            # lower front rail
RIM_T, RIM_H = 12.0 * MM, 20.0 * MM   # stiffening rim along the top of the front rail (box section)
WALL = 2.5 * MM                # moulded polystyrene walls and floor
BIN_MASS = 0.45                # kg, empty
POCKET_X = 70.0 * MM           # pocket centre, from the bin's front face (ahead of the centre of mass, x 0.094: it hangs and leans back on the liner)
POCKET_W = 22.0 * MM           # pocket width (x), inside
POCKET_H = 25.0 * MM           # pocket height (z), open at the bottom
POCKET_DEEP = 7.0 * MM         # pocket recess depth (y), from the end wall's outer face

# Door section (world, relative to the table top). The seated bin: front face at XF, bottom ZB0 above the table.
XF = 0.260
XP = XF + POCKET_X             # boss and pocket centre line, x = 0.330
XL = XF + BIN_D + 2.0 * MM     # liner face, 2 mm behind the seated bin's back wall (a bin leans back on it)
GAP = 2.0 * MM                 # side clearance between each bin end and its dike
Y_DIKE = BIN_L / 2 + GAP       # dike inner faces at y = +-0.2305 (461 mm apart)
DIKE_T = 12.0 * MM
DOOR_H = 200.0 * MM            # the lower 200 mm of the door
DOOR_T = 40.0 * MM             # liner, foam and outer skin
LEAD = 5.0 * MM                # rounded front inner edge of each dike (a 5 mm 45 deg chamfer here)
ZB0 = 15.0 * MM                # seated bin bottom above the bench (the lowest support position)
BOSS_W, BOSS_H, BOSS_OUT = 20.0 * MM, 12.0 * MM, 6.0 * MM
BOSS_CHAMFER = 4.0 * MM        # tapered top, 12 mm flat
CAP_W, CAP_Y, CAP_T = 10.0 * MM, 5.0 * MM, 3.0 * MM
FIT = 4.0 * MM                 # the snug last 4 mm of travel (press-fit cap travel)
FIT_FORCE = (4.0, 3.5)         # N static / dynamic per end, an estimate
Z_BT = ZB0 + POCKET_H - CAP_T  # boss top, 0.037 above the table
Z_BB = Z_BT - BOSS_H
BOSS_Y_IN = Y_DIKE - BOSS_OUT  # boss inner face, |y| = 0.2245
CAP_YC = BOSS_Y_IN + 0.5 * MM + CAP_Y / 2

# Start: the bin on a 3 mm rubber mat in front of the door, its front rail toward the robot.
MAT_T = 3.0 * MM
XS = 0.045                     # start front face x, clear of the arm bases; its back (x 0.248) is 12 mm in front of the dikes

WHITE = (0.93, 0.93, 0.91)
CLEAR = (0.38, 0.60, 0.80)     # blue-tinted clear polystyrene
RAIL = (0.30, 0.50, 0.72)
DARK = (0.16, 0.16, 0.17)
PS = (0.40, 0.35)              # polystyrene
ABS = (0.35, 0.30)
RUBBER = (0.90, 0.80)


def _extrude(name, outline, axis, a0, a1, color=None):
    """Closed convex prism: a convex polygon in the plane normal to ``axis``, from ``a0`` to ``a1`` along it.
    Outline points are (u, v) = (x, y) for axis z, (x, z) for axis y."""
    prof = [tuple(map(float, p)) for p in outline]
    n = len(prof)

    def pt(u, v, w):
        return (u, v, w) if axis == "z" else (u, w, v)

    pts = np.array([pt(u, v, w) for w in (a0, a1) for (u, v) in prof], float)
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [(i, j, n + j), (i, n + j, n + i)]
    for k in range(1, n - 1):
        faces += [(0, k + 1, k), (n, n + k, n + k + 1)]
    faces = np.array(faces, int)
    tri = pts[faces]
    c = pts.mean(0)
    if np.einsum('ij,ij->i', tri[:, 0] - c, np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])).sum() < 0:
        faces = faces[:, ::-1]
    proxy = (box((pts.min(0) + pts.max(0)) / 2, np.ptp(pts, axis=0)),)
    return Mesh(name, pts, faces, collision="convex", color=color, proxy=proxy)


def _door():
    """Fixed door-liner section; part frame = world shifted to the table top."""
    width = 2 * (Y_DIKE + DIKE_T) + 0.060
    base = [box((XL + DOOR_T / 2, 0.0, DOOR_H / 2), (DOOR_T, width, DOOR_H), WHITE)]
    meshes = []
    for s, tag in ((1, "l"), (-1, "r")):
        yi, yo = s * Y_DIKE, s * (Y_DIKE + DIKE_T)
        meshes.append(_extrude(f"dike_{tag}", [(XF, yo), (XF, yi + s * LEAD), (XF + LEAD, yi), (XL + 1 * MM, yi),
                                               (XL + 1 * MM, yo)], "z", 0.0, DOOR_H, WHITE))
        x0, x1 = XP - BOSS_W / 2, XP + BOSS_W / 2
        boss = [(x0, Z_BB), (x1, Z_BB), (x1, Z_BT - BOSS_CHAMFER), (x1 - BOSS_CHAMFER, Z_BT),
                (x0 + BOSS_CHAMFER, Z_BT), (x0, Z_BT - BOSS_CHAMFER)]
        y0, y1 = sorted((s * BOSS_Y_IN, s * (Y_DIKE + 0.5 * MM)))
        meshes.append(_extrude(f"boss_{tag}", boss, "y", y0, y1, WHITE))
    links = []
    for s, tag in ((1, "l"), (-1, "r")):
        c = (XP, s * CAP_YC, Z_BT + FIT + CAP_T / 2)
        links.append(Link(f"cap_{tag}", [box(c, (CAP_W, CAP_Y, CAP_T))],
                          Slide("z", c, -FIT, 0.0, damping=5.0, armature=0.01,
                                static_friction=FIT_FORCE[0], dynamic_friction=FIT_FORCE[1]),
                          mass=0.002, color=WHITE, friction=PS))
    return MeshMechanism("door", (0.0, 0.0, TOP), base, links, mass=None, color=WHITE, friction=ABS,
                         contact_offset=0.001, meshes={"base": tuple(meshes)})


def _bin_shapes():
    L2, t = BIN_L / 2, WALL
    px0, px1 = POCKET_X - POCKET_W / 2, POCKET_X + POCKET_W / 2
    yb = L2 - POCKET_DEEP              # |y| of the pocket back plate's outer (boss-side) face
    shapes = []
    # Front rail with its stiffening rim along the top edge, split so each hand grips its own 40 mm segment
    # centred at y = +-0.150.
    for y0, y1 in ((-0.170, -0.130), (0.130, 0.170), (-L2, -0.170), (-0.130, 0.130), (0.170, L2)):
        yc, wy = (y0 + y1) / 2, y1 - y0
        shapes.append(box((RIM_T / 2, yc, FRONT_H - RIM_H / 2), (RIM_T, wy, RIM_H), RAIL))
        shapes.append(box((t / 2, yc, (FRONT_H - RIM_H) / 2), (t, wy, FRONT_H - RIM_H), RAIL))
    shapes.append(box((BIN_D - t / 2, 0.0, BIN_H / 2), (t, BIN_L, BIN_H), CLEAR))          # back wall
    shapes.append(box((BIN_D / 2, 0.0, t / 2), (BIN_D - 2 * t, 2 * (yb - t), t), CLEAR))     # floor
    for s in (1, -1):
        yw = s * (L2 - t / 2)
        # end wall around the pocket opening
        shapes.append(box((px0 / 2, yw, BIN_H / 2), (px0, t, BIN_H), CLEAR))
        shapes.append(box(((px1 + BIN_D) / 2, yw, BIN_H / 2), (BIN_D - px1, t, BIN_H), CLEAR))
        shapes.append(box((POCKET_X, yw, (POCKET_H + BIN_H) / 2), (POCKET_W, t, BIN_H - POCKET_H), CLEAR))
        # floor strips in front of and behind the pocket, out to the end wall
        ys, wy = s * (yb - t + (L2 - yb) / 2), L2 - yb
        shapes.append(box(((t + px0 - t) / 2, ys, t / 2), (px0 - 2 * t, wy, t), CLEAR))
        shapes.append(box(((px1 + t + BIN_D - t) / 2, ys, t / 2), (BIN_D - px1 - 2 * t, wy, t), CLEAR))
        # pocket: back plate, two side plates and roof; the recess opens downward and outward
        shapes.append(box((POCKET_X, s * (yb - t / 2), (POCKET_H + t) / 2), (POCKET_W + 2 * t, t, POCKET_H + t), CLEAR))
        for xs in (px0 - t / 2, px1 + t / 2):
            shapes.append(box((xs, s * (yb + L2 - t) / 2, (POCKET_H + t) / 2), (t, L2 - t - yb, POCKET_H + t), CLEAR))
        shapes.append(box((POCKET_X, s * (yb + L2 - t) / 2, POCKET_H + t / 2), (POCKET_W + 2 * t, L2 - t - yb, t), CLEAR))
    return shapes


DOOR = _door()
MAT = Part("mat", (0.140, 0.0, TOP), [box((0.0, 0.0, MAT_T / 2), (0.230, 0.464, MAT_T), DARK)],
           mass=None, color=DARK, friction=RUBBER, contact_offset=0.001)
BIN = Part("bin", (XS, 0.0, TOP + MAT_T), _bin_shapes(), mass=BIN_MASS, color=CLEAR, friction=PS,
           contact_offset=0.002)
OBJECTS = [DOOR, MAT, BIN]
PRESSED_PARTS = ("door",)

# ------------------------------------------------------------------ scoring
ENGAGE_IN = 5.0 * MM           # engaged: the boss top at least 5 mm up inside the pocket
SEAT_TOL = 2.0 * MM            # locked: the pocket roof within 2 mm of the cap top, the cap within 0.5 mm of home
HOME_TOL = 0.5 * MM
LEVEL_DEG = 3.0              # a hung bin leans back about 1 deg onto the liner
DROP_V = 0.9                   # m/s downward: a free fall from 41 mm or more
NEAR_LINER = 25.0 * MM     # 0.25 tier: carried in over the bosses, back within 25 mm of the liner
TEST_STICKY = ()


def sticky(key, cond):
    return ever(key, bool(cond) or key in TEST_STICKY)


def boss_top(s):
    """World point at the centre of the boss's top flat (s = +1 left, -1 right)."""
    return np.array([XP, s * (BOSS_Y_IN + BOSS_OUT / 2), TOP + Z_BT])


def to_bin(b, p):
    return b.rotation().T @ (np.asarray(p, float) - np.asarray(b.origin, float))


def end_state(b, door, s):
    tag = "l" if s > 0 else "r"
    p = to_bin(b, boss_top(s))
    inside_x = abs(p[0] - POCKET_X) <= POCKET_W / 2
    inside_y = BIN_L / 2 - POCKET_DEEP - 1.0 * MM <= s * p[1] <= BIN_L / 2 + 4.0 * MM
    engaged = bool(inside_x and inside_y and ENGAGE_IN <= p[2] <= POCKET_H + 1.0 * MM)
    q = float(door.joints[f"cap_{tag}"])
    cap_top = to_bin(b, (XP, s * CAP_YC, TOP + Z_BT + FIT + CAP_T + q))
    home = q <= -FIT + HOME_TOL
    locked = bool(engaged and home and abs(POCKET_H - cap_top[2]) <= SEAT_TOL)
    return engaged, locked


def tilt_deg(b):
    return math.degrees(math.acos(float(np.clip(b.rotation()[2, 2], -1.0, 1.0))))


def between_dikes(b):
    low, high = np.asarray(b.low), np.asarray(b.high)
    return bool(low[1] >= -Y_DIKE - 1.0 * MM and high[1] <= Y_DIKE + 1.0 * MM and high[0] >= XL - NEAR_LINER
                and low[2] >= TOP + 5.0 * MM)


STEP_M, STEP_DEG = 0.0005, 0.25   # still: under 0.5 mm and 0.25 deg per 50 ms step (1 cm/s, 5 deg/s)
_POSE = {"episode": None, "token": None, "prev": None, "quiet": False, "drop": False}


def motion(b, door):
    """(quiet, dropping) from pose differences between consecutive control steps.

    Contact chatter makes the bin's reported velocities unreliable while it rests on the thin-walled mat contacts,
    so stillness and falling are judged from its corner points and the caps' positions step to step."""
    corners = np.array([[x, y, z] for x in (0.0, BIN_D) for y in (-BIN_L / 2, BIN_L / 2) for z in (0.0, BIN_H)])
    pts = np.asarray(b.origin, float) + corners @ b.rotation().T
    caps = np.array([door.joints["cap_l"], door.joints["cap_r"]], float)
    if _RUN["t"] is None:
        return bool(b.resting() and door.still(0.01, 0.05)), bool(float(b.lin_vel[2]) < -DROP_V)
    if _POSE["episode"] != _RUN["episode"]:
        _POSE.update(episode=_RUN["episode"], token=None, prev=None, quiet=False, drop=False)
    token = (_RUN["step"], float(_RUN["t"]))
    if token != _POSE["token"]:
        quiet = drop = False
        if _POSE["prev"] is not None:
            step, (old_pts, old_caps) = _POSE["prev"]
            if token[0] == step + 1:
                quiet = bool(np.all(np.linalg.norm(pts - old_pts, axis=1) < STEP_M + math.radians(STEP_DEG) * 0.25)
                             and np.linalg.norm(pts.mean(0) - old_pts.mean(0)) < STEP_M
                             and np.all(np.abs(caps - old_caps) < 0.0002))
                drop = bool(old_pts[:, 2].mean() - pts[:, 2].mean() > DROP_V * 0.05)
        _POSE.update(prev=(token[0], (pts, caps)), token=token, quiet=quiet, drop=drop)
    return _POSE["quiet"], _POSE["drop"]


def score(objects):
    b, door = objects["bin"], objects["door"]
    quiet, falling = motion(b, door)
    dropped = sticky("dropped", falling)
    upset = sticky("upset", b.rotation()[2, 2] < 0.0 or float(b.origin[2]) < TOP - 0.05)
    rest = released(b) and quiet
    seconds = held_for("released and still", rest)
    if dropped or upset:
        return 0.0
    if not (rest and seconds + 1e-9 >= STILL_S):
        return 0.0
    ends = [end_state(b, door, s) for s in (1, -1)]
    n_eng = sum(e for e, _ in ends)
    n_lock = sum(l for _, l in ends)
    if n_lock == 2 and tilt_deg(b) < LEVEL_DEG:
        return 1.0
    if n_eng == 2:
        return 0.75
    if n_eng == 1:
        return 0.5
    if between_dikes(b):
        return 0.25
    return 0.0


def success(objects):
    return score(objects) >= 1.0


# ------------------------------------------------------------------ scripted demo (start layout and time only)
TIP = 0.013                    # pad tips below the TCP
L_YAW, R_YAW = -math.pi / 2, math.pi / 2   # pads close along x, across the front rail
GRIP_Y = 0.150                 # the hands 300 mm apart on the front rail
GRIP_DEPTH = 15.0 * MM         # pad tips this far below the rim top (the rim is 20 mm tall)
CARRY_UP = 0.069               # bin bottom lifted from 3 mm to 72 mm: 28 mm over the cap tops
RIDE_LOW = 4.0 * MM            # a carried part rides low in the grip
PUSH_PAST = 10.0 * MM          # commanded past the seated height: a firm press
CARRY_LEAD = 8.0 * MM          # the loaded arms lag about 8 mm at full reach (GPU fb-06): lead the carry
DEMO_PACE = 1.0


def demo():
    plan = {"left": [], "right": []}
    gx0 = XS + RIM_T / 2
    gx1 = XF + RIM_T / 2
    z_grip = MAT_T + FRONT_H - GRIP_DEPTH + TIP          # TCP height above the table at the grasp
    z_carry = z_grip + CARRY_UP + RIDE_LOW
    z_touch = z_grip + (ZB0 + FIT) - MAT_T                # pocket roofs on the caps
    z_home = z_grip + ZB0 - MAT_T                         # seated home
    steps = [
        # (seconds, label, x, z above the table, finger)
        (2.0, "over the front rail", gx0, z_grip + 0.060, FINGER_OPEN),
        (1.2, "pads around the rail", gx0, z_grip, FINGER_OPEN),
        (0.8, "grip", gx0, z_grip, FINGER_CLOSED),
        (1.6, "lift level", gx0, z_carry, FINGER_CLOSED),
        (3.5, "carry in above the bosses", gx1 + CARRY_LEAD, z_carry, FINGER_CLOSED),
        (0.6, "settle", gx1 + CARRY_LEAD, z_carry, FINGER_CLOSED),
        (2.0, "lower both ends together", gx1 + CARRY_LEAD / 2, z_touch + RIDE_LOW + 0.004, FINGER_CLOSED),
        (1.0, "onto the bosses", gx1, z_touch, FINGER_CLOSED),
        (1.0, "push down", gx1, z_home - PUSH_PAST, FINGER_CLOSED),
        (0.8, "hold", gx1, z_home - PUSH_PAST, FINGER_CLOSED),
        (0.6, "ease off", gx1, z_home, FINGER_CLOSED),
        (0.8, "let go", gx1, z_home, FINGER_OPEN),
        (0.8, "rise", gx1, z_home + 0.030, FINGER_OPEN),
        (1.2, "back away", 0.120, 0.160, FINGER_OPEN),
        (4.0, "stillness tail", 0.120, 0.160, FINGER_OPEN),
    ]
    for dt, label, x, z, finger in steps:
        plan["left"].append(Waypoint(dt * DEMO_PACE, (float(x), GRIP_Y, float(TOP + z)), finger, label, L_YAW))
        plan["right"].append(Waypoint(dt * DEMO_PACE, (float(x), -GRIP_Y, float(TOP + z)), finger, label, R_YAW))
    return plan


def demo_duration():
    return sum(w.duration for w in demo()["right"])
