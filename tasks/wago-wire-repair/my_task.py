"""Repair a shallow wire connection in a bench-mounted two-port lever terminal.

Use the left arm's permanently mounted narrow opening tool to raise the orange
lever. With the right gripper, withdraw the blue solid-core lead, insert its
11 mm stripped end fully, release it, close the lever and withdraw both hands.
The other port is a fixed reference lead. This is a cold, low-voltage training
approximation; green indicates geometric continuity, not a circuit simulation.
"""
import importlib.abc
import importlib.machinery
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from yam_arena.task_file import Asset, TABLE_TOP
from yam_arena.demo import Waypoint
from yam_arena.urdf import quat_xyzw_to_matrix

HERE = Path(__file__).resolve().parent
SCENE = 'station'
SUCCESS_HOLD_S = 2.
SCORE_OVER = 'end'
BASE = np.array([.25, -.16, TABLE_TOP[SCENE] + .035])
PIVOT = BASE + [.00325, .005, .0043]
WIRE_START = BASE + [.00325, -.04115, .0037]
WIRE_GOAL = BASE + [.00325, -.03315, .0037]
GRIP_OFFSET = np.array([0.,-.025,.006])
HISTORY = {'opened': False, 'withdrawn': False}
FEEDBACK = {}

@dataclass(frozen=True)
class Model(Asset):
    asset: str = 'local_lever_terminal'
    xy: tuple = (0., 0.)
    extent: float = 1.
    pos: tuple = (0., 0., 0.)
    @property
    def free(self):
        return self.name == 'wire'

OBJECTS = [Model('fixture', pos=tuple(BASE)),
           Model('lever', mass=.0004, pos=tuple(PIVOT)),
           Model('clamp', mass=.0002, pos=tuple(BASE + [.00325, 0., .0032])),
           Model('wire', mass=.002, pos=tuple(WIRE_START))]

def measurements(objects):
    w = objects['wire']
    axis = quat_xyzw_to_matrix(w.quat_xyzw)[:, 1]
    # Runtime supplies the rigid-body origin: the asymmetric wire AABB centre is not the grasp point.
    tip = np.asarray(objects.get('_wire_origin', w.pos)) + axis * .035
    depth = tip[1] - (BASE[1] - .00915)
    lateral = np.linalg.norm((tip - (BASE + [.00325, 0., .0037]))[[0, 2]])
    q = objects['lever'].quat_xyzw
    angle = -math.degrees(math.atan2(2*(q[3]*q[0]+q[1]*q[2]), 1-2*(q[0]**2+q[1]**2)))
    seated = .0103 <= depth <= .012 and lateral < .0008 and axis[1] > math.cos(math.radians(6))
    return depth, lateral, angle, bool(seated)

def distance(objects):
    return float(np.linalg.norm(np.asarray(objects.get('_wire_origin',objects['wire'].pos)) - WIRE_GOAL))

def wire_resting(objects):
    # Runtime checks 0.5 seconds of position/axis stability, ignoring axial spin.
    return bool(objects.get('_wire_stable', False))

def success(objects):
    _, _, angle, seated = measurements(objects)
    return bool(seated and abs(angle) < 5 and objects.get('_opened', False)
                and objects.get('_withdrawn', False) and objects.get('_clamped', False)
                and objects.get('_hands_clear', False)
                and wire_resting(objects))

def score(objects):
    depth, lateral, angle, seated = measurements(objects)
    if objects['wire'].pos[2] < TABLE_TOP[SCENE]:
        return 0.
    return float(.15 * objects.get('_opened', False) + .15 * objects.get('_withdrawn', False)
                 + .35 * seated + .25 * (seated and abs(angle) < 5 and objects.get('_clamped', False)))

def tab(angle):
    a = math.radians(-angle)
    y, z = -.012, .0032
    return PIVOT + [0., math.cos(a)*y-math.sin(a)*z, math.sin(a)*y+math.cos(a)*z]

@dataclass
class PrecisionWaypoint(Waypoint):
    pass

def tool_tcp(angle, closing=False):
    # Tangent support of the rectangular blade against the rotating lever face.
    a = math.radians(angle)
    normal = np.array([0., math.sin(a), math.cos(a)])
    y, z = -.010, .00385 if closing else .00255
    surface = PIVOT + [0., math.cos(a)*y+math.sin(a)*z,
                        -math.sin(a)*y+math.cos(a)*z]
    support = .0035*abs(normal[1]) + .0004*abs(normal[2]) + .0001
    centre = surface + (1. if closing else -1.)*normal*support
    return centre + [0., .076, .04]

def demo():
    """Hold the lever open with the left tool during right-hand wire repair."""
    HISTORY.update(opened=False, withdrawn=False)
    FEEDBACK.clear()
    left, right = [], []
    def l(t,p,label): left.append(PrecisionWaypoint(t,tuple(p),0.,label))
    def r(t,p,finger,label): right.append(PrecisionWaypoint(t,tuple(p),finger,label,yaw=-math.pi/2))
    def wait_to(points,time,label):
        duration = time-sum(w.duration for w in points)
        if duration>1e-6:
            w = points[-1]
            points.append(PrecisionWaypoint(duration,w.pos,w.finger,label,yaw=w.yaw))
    clear = np.array([.20,.20,BASE[2]+.18])
    park = np.array([.20,-.24,BASE[2]+.15])
    low = tool_tcp(0)+[0.,-.003,0.]
    l(3,low+[0.,.025,.035],'approach lever')
    l(2,low+[0.,-.01,0.],'approach below tab')
    l(2,low,'engage underside')
    l(4,low,'settle below tab')
    for angle in range(0,101,10): l(.5,tool_tcp(angle),'lift lever')
    l(.5,tool_tcp(100),'hold lever open')
    r(3,park,-.012,'wait away from tool')
    # Reach the hover as opening finishes; work on the wire while the left
    # tool remains engaged. Both arms run on the same simulation clock.
    wait_to(right,sum(w.duration for w in left)-2.,'wait for lever opening')
    grip = WIRE_START+GRIP_OFFSET
    insertion = WIRE_GOAL+[0.,0.,.00025]
    withdrawn = grip+[0.,-.016,0.]
    r(2,grip+[0.,0.,.08],-.012,'approach lead')
    r(2,grip,-.012,'descend around insulation')
    r(1.5,grip,-.012,'settle around insulation')
    r(1.5,grip,0.,'pinch insulation')
    r(2,withdrawn,0.,'withdraw stripped end')
    r(1,withdrawn,0.,'hold outside mouth')
    r(4,insertion+GRIP_OFFSET,0.,'insert to insulation stop')
    r(1,insertion+GRIP_OFFSET,0.,'hold inserted wire')
    r(1,insertion+GRIP_OFFSET,-.006,'release wire')
    r(.5,insertion+GRIP_OFFSET,-.006,'release wire')
    wait_to(left,sum(w.duration for w in right),'hold lever open')
    # After wire release, both arms retreat concurrently before closure.
    l(2,clear,'clear open lever')
    r(1.5,park,-.006,'park right arm')
    # Push the lever's rear face past vertical with the vertical tool stem.
    # Once over centre the passive spring closes it; no lever motor is used.
    push_start = BASE+[.00325,.093,.047]
    push_end = BASE+[.00325,.085,.047]
    l(1.5,push_start+[0.,.012,.025],'approach behind lever')
    l(1.5,push_start,'descend behind lever')
    l(3,push_end,'nudge lever closed')
    l(1,push_end,'wait for clamp')
    l(2,tab(0)+[0.,.12,.1],'withdraw tool')
    l(2,tab(0)+[0.,.12,.1],'hold clear')
    wait_to(right,sum(w.duration for w in left),'park right arm')
    return {'left':left,'right':right}

def _attach(scene):
    if getattr(scene.task_object, '_wago_local', False):
        return
    name = '_wago_runtime_' + str(abs(hash(str(HERE))))
    spec = importlib.util.spec_from_file_location(name, HERE / 'mechanism.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    from types import SimpleNamespace
    module.install(scene, SimpleNamespace(**globals()))
    _install_tracking()

def _install_tracking():
    # Scripted-policy-only integral joint tracking for the native compliant servos.
    # Actual joint measurements affect robot actions, never object poses or forces.
    from yam_arena.demo import ArmPlan
    from yam_arena import ik
    from yam_arena.urdf import axis_angle_matrix
    native_target = ArmPlan.target
    def target(self, t):
        p,r,finger,label = native_target(self,t)
        if not self.waypoints or not isinstance(self.waypoints[0],PrecisionWaypoint):
            return p,r,finger,label
        if self.base_pos[1]>0:
            angle = FEEDBACK.get('lever_angle',0.)
            if label in ('lift lever','hold lever open'):
                p = tool_tcp(float(np.clip(angle+5.,0.,100.)))
            return p,r,finger,label
        hand = FEEDBACK.get('right_robot')
        wire = FEEDBACK.get('wire_origin')
        if label == 'withdraw stripped end' and not hasattr(self,'_wire_grasp') and hand is not None and wire is not None:
            if np.linalg.norm(wire-hand[0]) < .050:
                self._wire_grasp = (hand[1].T @ (wire-hand[0]),hand[1].T @ FEEDBACK['wire_axis'])
        if hasattr(self,'_wire_grasp') and label in (
            'withdraw stripped end','hold outside mouth','insert to insulation stop',
            'hold while lever closes','hold inserted wire','release wire','withdraw hand'):
            relative, axis = self._wire_grasp
            carried_axis = r @ axis
            cross = np.cross(carried_axis,np.array([0.,1.,0.]))
            sine = np.linalg.norm(cross)
            if sine>1e-8:
                correction = axis_angle_matrix(cross/sine,math.atan2(sine,float(carried_axis[1])))
                r = correction @ r
            desired_root = p - GRIP_OFFSET
            p = desired_root - r @ relative
            if label in ('hold outside mouth','insert to insulation stop','hold while lever closes','hold inserted wire') and wire is not None:
                tip = wire + FEEDBACK['wire_axis']*.035
                wanted_tip = desired_root + [0.,.035,0.]
                error = wanted_tip-tip
                bias = getattr(self,'_wire_tip_bias',np.zeros(3))
                bias[[0,2]] = np.clip(bias[[0,2]] + .05*error[[0,2]],-.004,.004)
                self._wire_tip_bias = bias
                correction = bias.copy()
                correction[[0,2]] += .8*error[[0,2]]
                self._wire_last_correction = np.clip(correction,-.005,.005)
                p += self._wire_last_correction
            elif label in ('release wire','withdraw hand','done'):
                p += getattr(self,'_wire_last_correction',np.zeros(3))
        return p,r,finger,label
    ArmPlan.target = target
    original = ArmPlan.action7
    if getattr(original, '_wago_tracking', False): return
    def action7(self, t):
        action, label = original(self, t)
        if not self.waypoints or not isinstance(self.waypoints[0], PrecisionWaypoint):
            return action, label
        side = 'left_robot' if self.base_pos[1] > 0 else 'right_robot'
        feedback = FEEDBACK.get(side)
        integral = getattr(self, '_wago_integral', np.zeros(6))
        if feedback is not None:
            integral = np.clip(integral + (.12 if side == 'left_robot' else .04)*(action[:6]-feedback[2]), -.5, .5)
        self._wago_integral = integral
        FEEDBACK.setdefault('tracking',{})[side] = integral.tolist()
        action[:6] = np.clip(action[:6]+integral,ik.LOWER,ik.UPPER)
        return action, label
    action7._wago_tracking = True
    ArmPlan.action7 = action7

class _Loader(importlib.abc.Loader):
    def __init__(self, original): self.original = original
    def create_module(self, spec): return self.original.create_module(spec)
    def exec_module(self, module):
        self.original.exec_module(module)
        _attach(module)

class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'yam_arena.scene':
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.loader:
                spec.loader = _Loader(spec.loader)
            return spec

if 'yam_arena.scene' in sys.modules:
    _attach(sys.modules['yam_arena.scene'])
else:
    sys.meta_path.insert(0, _Finder())
