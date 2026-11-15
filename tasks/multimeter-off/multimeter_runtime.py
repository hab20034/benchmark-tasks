"""Simulation-only implementation of the multimeter submission's custom assets.

Loaded by multimeter_addon only after Isaac Sim starts. The adapter delegates ordinary
objects and tasks to their native factories. It does not replace the policy, scoring,
observations, or termination logic, and never changes files in the installed environment.
"""

import json
import math
import os
from dataclasses import fields
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg
from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.utils import clone
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.utils.cameras import ArenaCameraCfg
from isaaclab_arena.utils.configclass import make_configclass
from isaaclab_arena.utils.pose import Pose
from isaaclab_physx.sim.schemas import PhysxCollisionPropertiesCfg


def _yaw_xyzw(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def make_object(spec) -> Object:
    """A local model, optionally attached to a passive world-mounted hinge."""
    from pxr import Gf, Usd, UsdPhysics

    quat = _yaw_xyzw(spec.yaw)

    @clone
    def spawn(prim_path, cfg, *args, **kwargs):
        prim = spawn_from_usd.__wrapped__(prim_path, cfg, *args, **kwargs)
        if spec.mass is None:
            for part in Usd.PrimRange(prim):
                if part.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(part).CreateKinematicEnabledAttr(True)
        if spec.collision_filter:
            UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().SetTargets([
                prim.GetParent().GetPath().AppendChild(name) for name in spec.collision_filter
            ])
        if spec.hinge is not None:
            pivot = tuple(p * s for p, s in zip(spec.hinge.pivot, spec.scale, strict=True))
            x, y, z = pivot
            c, s = math.cos(spec.yaw), math.sin(spec.yaw)
            anchor = (spec.pos[0] + c * x - s * y, spec.pos[1] + s * x + c * y, spec.pos[2] + z)
            joint = UsdPhysics.RevoluteJoint.Define(prim.GetStage(), prim_path + "/hinge")
            joint.CreateBody1Rel().SetTargets([prim.GetPath()])
            joint.CreateAxisAttr("Z")
            joint.CreateLocalPos0Attr(Gf.Vec3f(*anchor))
            joint.CreateLocalPos1Attr(Gf.Vec3f(*pivot))
            joint.CreateLocalRot0Attr(Gf.Quatf(quat[3], Gf.Vec3f(*quat[:3])))
            joint.CreateLowerLimitAttr(spec.hinge.lower_degrees)
            joint.CreateUpperLimitAttr(spec.hinge.upper_degrees)
            joint.CreateCollisionEnabledAttr(False)
        return prim

    addon = {
        "func": spawn,
        "collision_props": PhysxCollisionPropertiesCfg(contact_offset=0.0002, rest_offset=0.0),
        "physics_material": sim_utils.RigidBodyMaterialCfg(
            static_friction=spec.friction[0], dynamic_friction=spec.friction[1]
        ),
    }
    if spec.mass is not None:
        addon.update(mass_props={"": [sim_utils.MassCfg(mass=spec.mass)]}, mass_props_create_if_missing=True)
    return Object(
        name=spec.name, prim_path="{ENV_REGEX_NS}/" + spec.name,
        object_type=ObjectType.RIGID if spec.mass is not None else ObjectType.BASE,
        usd_path=spec.usd_path, scale=tuple(float(s) for s in spec.scale),
        initial_pose=Pose(position_xyz=tuple(float(p) for p in spec.pos), rotation_xyzw=quat), spawn_cfg_addon=addon,
    )


def _reset_hinges(env, env_ids, specs):
    """Reset ratchets together with object poses, including repeated seeded episodes."""
    from pxr import UsdPhysics

    ids = range(env.num_envs) if env_ids is None else env_ids
    for index in ids:
        for spec in specs:
            path = env.scene.env_prim_paths[int(index)] + f"/{spec['name']}/hinge"
            UsdPhysics.RevoluteJoint.Get(env.sim.stage, path).GetLowerLimitAttr().Set(spec["lower"])


def _ratchet_hinges(env, env_ids, specs):
    """Advance only the lower stop; finger contact supplies the turning torque."""
    from pxr import UsdPhysics

    ids = range(env.num_envs) if env_ids is None else env_ids
    for spec in specs:
        if not spec["ccw"]:
            continue
        quats = env.scene[spec["name"]].data.root_quat_w.torch.detach().cpu().numpy()
        for index in ids:
            x, y, z, w = quats[int(index)]
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            angle = math.degrees(math.atan2(math.sin(yaw - spec["yaw"]), math.cos(yaw - spec["yaw"])))
            path = env.scene.env_prim_paths[int(index)] + f"/{spec['name']}/hinge"
            lower = UsdPhysics.RevoluteJoint.Get(env.sim.stage, path).GetLowerLimitAttr()
            lower.Set(float(max(lower.Get(), min(spec["upper"], angle - 0.5))))


def install(scene_module, local_type):
    """Add support for LocalUsdAsset instances only, with no on-disk environment changes."""
    native_object = scene_module.task_object
    native_build = scene_module.build_task_environment

    def task_object(spec):
        if isinstance(spec, local_type):
            return make_object(spec)
        return native_object(spec)

    def build_task_environment(embodiment, task, control_hz=None):
        local = [obj for obj in task.objects if isinstance(obj, local_type)]
        names = set(task.names)
        for obj in local:
            if set(obj.collision_filter) - (names - {obj.name}):
                raise ValueError(f"{obj.name}.collision_filter must name other task objects")
        if local:
            right = embodiment.scene_config.right_robot
            right.spawn = right.spawn.replace(
                collision_props=PhysxCollisionPropertiesCfg(contact_offset=0.0002, rest_offset=0.0)
            )
            sensors = []
            for finger in ("tip_left", "tip_right"):
                sensors.append(("dial_" + finger, ContactSensorCfg, ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/RightArm" + embodiment.link_paths[finger],
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/dial"],
                    update_period=0.0, track_contact_points=True, history_length=1,
                    max_contact_data_count_per_prim=128,
                )))
            existing = [(f.name, f.type, getattr(embodiment.scene_config, f.name))
                        for f in fields(embodiment.scene_config)]
            embodiment.scene_config = make_configclass("MultimeterPinchSceneCfg", existing + sensors)()
            if os.environ.get("MULTIMETER_PINCH_LOG"):
                import numpy as np
                from yam_arena.ik import matrix_to_quat_xyzw

                pivot = np.asarray(task.module.PIVOT_WORLD)
                eye = pivot + (0.12, 0.015, 0.08)
                forward = (pivot - eye) / np.linalg.norm(pivot - eye)
                right_axis = np.cross(forward, [0.0, 0.0, 1.0])
                right_axis /= np.linalg.norm(right_axis)
                up = np.cross(right_axis, forward)
                quat = matrix_to_quat_xyzw(np.column_stack((right_axis, up, -forward)))
                camera = CameraCfg(
                    prim_path="{ENV_REGEX_NS}/PinchCamera", height=480, width=640, data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.005, 5.0)),
                    offset=CameraCfg.OffsetCfg(pos=tuple(float(v) for v in eye),
                                              rot=tuple(float(v) for v in quat), convention="opengl"),
                    renderer_cfg=embodiment.camera_config.top_cam.renderer_cfg,
                )
                cameras = [(f.name, f.type, getattr(embodiment.camera_config, f.name))
                           for f in fields(embodiment.camera_config)]
                embodiment.camera_config = make_configclass(
                    "MultimeterPinchCameraCfg", cameras + [("pinch_cam", CameraCfg, camera)], bases=(ArenaCameraCfg,)
                )()
        definition = native_build(embodiment, task, control_hz)
        hinges = [
            {"name": obj.name, "yaw": obj.yaw, "lower": obj.hinge.lower_degrees,
             "upper": obj.hinge.upper_degrees, "ccw": obj.hinge.counter_clockwise_only}
            for obj in local if obj.hinge is not None
        ]
        if hinges:
            native_configure = definition.env_cfg_callback

            def configure(cfg):
                cfg = native_configure(cfg)
                cfg.events.multimeter_hinges_reset = EventTermCfg(
                    func=_reset_hinges, mode="reset", params={"specs": hinges}
                )
                cfg.events.multimeter_hinges_ratchet = EventTermCfg(
                    func=_ratchet_hinges, mode="interval", interval_range_s=(0.0, 0.0),
                    params={"specs": hinges},
                )
                if os.environ.get("MULTIMETER_PINCH_LOG"):
                    cfg.events.multimeter_pinch_diagnostics = EventTermCfg(
                        func=_pinch_diagnostics, mode="interval", interval_range_s=(0.0, 0.0),
                        params={"out": os.environ["MULTIMETER_PINCH_LOG"]},
                    )
                return cfg

            definition.env_cfg_callback = configure
        return definition

    scene_module.task_object = task_object
    scene_module.build_task_environment = build_task_environment


def _pinch_diagnostics(env, env_ids, out):
    """Read filtered contacts only; never command robot, dial, or grasp attachment."""
    import numpy as np
    from yam_arena.runtime import tcp_world
    from yam_arena.urdf import quat_xyzw_to_matrix

    def array(value):
        value = value.torch if hasattr(value, "torch") else value
        return value.detach().cpu().numpy()

    pose = array(env.scene["dial"].data.root_pose_w)[0]
    rotation = quat_xyzw_to_matrix(pose[3:])
    pivot_local = np.array([-0.00899158977, 0.000009714819498, 0.03652296215])
    pivot = pose[:3] + rotation @ pivot_local
    robot = env.scene["right_robot"]
    ids = robot.find_joints(["joint7", "joint8"], preserve_order=True)[0]
    row = {
        "step": int(env.episode_length_buf[0]), "time": float(env.episode_length_buf[0]) * env.step_dt,
        "gap": float(-array(robot.data.joint_pos)[0, ids].sum()),
        "tcp_relative": (rotation.T @ (tcp_world(robot) - pivot)).tolist(),
        "dial_degrees": math.degrees(math.atan2(
            math.sin(math.atan2(rotation[1, 0], rotation[0, 0]) - math.pi),
            math.cos(math.atan2(rotation[1, 0], rotation[0, 0]) - math.pi),
        )),
    }
    for finger in ("tip_left", "tip_right"):
        data = env.scene.sensors["dial_" + finger].data
        force = array(data.normal_force_matrix_w)[0].reshape(-1, 3).sum(0)
        point = array(data.contact_pos_w)[0].reshape(-1, 3)[0]
        row[finger] = {
            "force": force.tolist(), "force_local": (rotation.T @ force).tolist(),
            "point_local": (rotation.T @ (point - pivot)).tolist(),
        }
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as log:
        log.write(json.dumps(row) + "\n")
    if row["step"] % 20 == 0:
        from PIL import Image
        image = array(env.scene.sensors["pinch_cam"].data.output["rgb"])[0, ..., :3]
        frames = path.parent / (path.stem + "_frames")
        frames.mkdir(exist_ok=True)
        Image.fromarray(image).save(frames / f"{row['step']:05d}.png")
