"""Original compact geometry and deferred Isaac adapter. No manufacturer CAD.

Run this file directly to regenerate assets. An ideal passive cam constraint lifts a spring-loaded
prismatic jaw; the lever has no drive. The lead is always a
free rigid body. There are no scripted object forces or pose corrections.
"""
from pathlib import Path
import math
import numpy as np

HERE = Path(__file__).resolve().parent

def generate():
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    assets = HERE / 'assets'
    assets.mkdir(exist_ok=True)
    grey, orange, copper = (.65,.69,.72), (1.,.28,.025), (.72,.38,.16)
    for name in ('fixture', 'lever', 'clamp', 'wire', 'tool'):
        stage = Usd.Stage.CreateNew(str(assets / (name + '.usd')))
        UsdGeom.SetStageUpAxis(stage, 'Z')
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        root = UsdGeom.Xform.Define(stage, '/Model').GetPrim()
        stage.SetDefaultPrim(root)
        if name in ('lever', 'clamp', 'wire'):
            UsdPhysics.RigidBodyAPI.Apply(root)
            UsdPhysics.MassAPI.Apply(root).CreateMassAttr({'lever':.0004,'clamp':.0002,'wire':.002}[name])
        def box(label, pos, size, color=grey, collision=True):
            shape = UsdGeom.Cube.Define(stage, '/Model/' + label)
            shape.CreateSizeAttr(1.)
            shape.AddTranslateOp().Set(Gf.Vec3d(*pos))
            shape.AddScaleOp().Set(Gf.Vec3f(*size))
            shape.CreateDisplayColorAttr([color])
            if collision: UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            return shape
        def cylinder(label, pos, radius, length, color, axis='Y', collision=True):
            shape = UsdGeom.Cylinder.Define(stage, '/Model/' + label)
            shape.CreateAxisAttr(axis)
            shape.CreateRadiusAttr(radius)
            shape.CreateHeightAttr(length)
            shape.AddTranslateOp().Set(Gf.Vec3d(*pos))
            shape.CreateDisplayColorAttr([color])
            if collision: UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        def wire_collision(label, y0, y1, radius):
            # Analytic capsules avoid cooking submillimetre cylinder hulls.
            shape = UsdGeom.Capsule.Define(stage,'/Model/'+label)
            shape.CreateAxisAttr('Y')
            shape.CreateRadiusAttr(radius)
            shape.CreateHeightAttr(y1-y0-2*radius)
            shape.AddTranslateOp().Set(Gf.Vec3d(0.,(y0+y1)/2,0.))
            shape.CreatePurposeAttr('guide')
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        def mouth_wall(label, x0, x1, low0, low1, high0, high1):
            y0,y1 = -.00915,-.00765
            points = [(x0[0],y0,low0),(x0[1],y0,low0),(x1[1],y1,low1),(x1[0],y1,low1),
                      (x0[0],y0,high0),(x0[1],y0,high0),(x1[1],y1,high1),(x1[0],y1,high1)]
            faces = [(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
            shape = UsdGeom.Mesh.Define(stage,'/Model/'+label)
            shape.CreatePointsAttr([Gf.Vec3f(*v) for v in points])
            shape.CreateFaceVertexCountsAttr([4]*6)
            shape.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
            shape.CreateSubdivisionSchemeAttr('none')
            shape.CreateDisplayColorAttr([grey])
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr('convexHull')
        if name == 'fixture':
            box('base', (0,-.025,-.031), (.09,.19,.008), (.12,.17,.23))
            box('pedestal',(0,0,-.015),(.020,.025,.030))
            box('floor',(0,0,.0005),(.013,.0183,.001))
            for x in (-.00615,0,.00615):
                box('wall'+str(x).replace('-','m').replace('.','_'),(x,0,.003),(.0007,.0183,.006))
            # Hinge-end clearance: the handle sweeps below pivot height past 90 degrees.
            box('rear',(0,.00865,.0015),(.013,.001,.003))
            for i,x in enumerate((-.00325,.00325)):
                box(f'bus{i}',(x,.0015,.0027),(.004,.009,.0006),copper)
                span = (x-.00275,x+.00275)
                mouth_wall(f'mouth_floor{i}',span,span,.0006,.0006,.0020,.0030)
                mouth_wall(f'mouth_top{i}',span,span,.0054,.0050,.0058,.0058)
                mouth_wall(f'mouth_left{i}',(x-.0026,x-.0020),(x-.0026,x-.0014),.0006,.0006,.0058,.0058)
                mouth_wall(f'mouth_right{i}',(x+.0020,x+.0026),(x+.0014,x+.0026),.0006,.0006,.0058,.0058)
            cylinder('reference_front',(-.00325,-.01125,.0037),.00135,.0035,(.2,.2,.2))
            cylinder('reference_bend',(-.019,-.013,.0037),.00135,.0315,(.2,.2,.2),'X')
            cylinder('reference_insulation',(-.035,-.038,.0037),.00135,.050,(.2,.2,.2))
            cylinder('reference_copper',(-.00325,-.004,.0037),.0007,.011,copper)
            box('reference_lever',(-.00325,-.001,.007),(.0047,.012,.0023),orange)
            box('rear_wire_rest',(.00325,-.115,.00175),(.006,.004,.0012))
            box('indicator_stand',(.026,.025,-.015),(.018,.02,.03))
            box('continuity_indicator',(.026,.025,.001),(.012,.012,.003),(.8,.05,.03),False)
        elif name == 'lever':
            box('handle',(0,-.006,.0032),(.0047,.012,.0013),orange)
            cylinder('pivot',(0,0,0),.0012,.0045,orange,'X',False)
            box('cam',(0,-.0025,0),(.0034,.0011,.0011),(.22,.24,.26),False)
        elif name == 'clamp':
            box('jaw',(0,.0005,.0005),(.0035,.005,.0014))
            box('follower',(0,.0025,.00185),(.0035,.009,.0004))
            box('neck',(0,-.001,.0008),(.0035,.0006,.0016))
        elif name == 'wire':
            cylinder('insulation',(0,-.0305,0),.00135,.109,(.025,.24,.8),collision=False)
            cylinder('copper',(0,.0295,0),.0007,.011,copper,collision=False)
            wire_collision('insulation_collision',-.085,.024,.00135)
            wire_collision('copper_collision',.023,.035,.0007)
        else:
            # Coordinates in the native gripper link at yaw=0: world X=-local Y,
            # world Y=-local X, world Z=-local Z. TCP is local Z=.134.
            # Visible external collar wraps the measured gripper housing
            # (bounds X=+-51, Y=+-32.5 mm). The bracket routes outside it.
            steel = (.24,.27,.30)
            for sign in (-1,1):
                box('collar_side_'+str(sign).replace('-','m'),
                    (sign*.054,0,.037),(.008,.080,.018),orange,False)
                box('collar_face_'+str(sign).replace('-','m'),
                    (0,sign*.036,.037),(.108,.008,.018),orange,False)
            box('side_bracket',(0,-.040,.082),(.010,.008,.098),steel,False)
            box('shaft_bridge',(0,-.020,.130),(.010,.046,.008),steel,False)
            box('stem',(0,0,.140),(.003,.003,.028))
            box('extension',(.038,0,.154),(.080,.003,.002))
            box('probe',(.079,0,.164),(.0015,.0015,.020))
            box('blade',(.076,0,.174),(.007,.0053,.0008))
            for sign in (-1,1):
                box('cheek_'+str(sign).replace('-','m'),(.076,sign*.00305,.1725),(.007,.0008,.003))
            box('hook_lip',(.073,0,.173),(.0008,.0053,.002))
        stage.GetRootLayer().Save()


def install(scene, task):
    from pxr import Gf, UsdGeom, UsdPhysics
    import isaaclab.sim as sim
    from isaaclab.sim.utils import clone
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_physx.sim.schemas import PhysxCollisionPropertiesCfg, PhysxRigidBodyPropertiesCfg
    from yam_arena.constants import TCP_OFFSET
    from yam_arena.urdf import quat_xyzw_to_matrix
    debug_sensors = {}
    native_object, native_build, native_states = scene.task_object, scene.build_task_environment, scene.object_states

    def task_object(spec):
        if not isinstance(spec, task.Model): return native_object(spec)
        @clone
        def spawn(path, cfg, *args, **kwargs):
            prim = spawn_from_usd.__wrapped__(path,cfg,*args,**kwargs)
            if spec.mass is not None:
                from pxr import PhysxSchema
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateEnableCCDAttr(True)
            if spec.name in ('lever','clamp'):
                cls = UsdPhysics.RevoluteJoint if spec.name == 'lever' else UsdPhysics.PrismaticJoint
                joint = cls.Define(prim.GetStage(), path+'/joint')
                joint.CreateBody1Rel().SetTargets([prim.GetPath()])
                joint.CreateAxisAttr('X' if spec.name == 'lever' else 'Z')
                joint.CreateLocalPos0Attr(Gf.Vec3f(*spec.pos))
                joint.CreateLocalPos1Attr(Gf.Vec3f(0,0,0))
                joint.CreateLowerLimitAttr(-100. if spec.name == 'lever' else 0.)
                joint.CreateUpperLimitAttr(0. if spec.name == 'lever' else .003)
                joint.CreateCollisionEnabledAttr(True)
                if spec.name == 'clamp':
                    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
                    drive.CreateTypeAttr('force')
                    drive.CreateTargetPositionAttr(-.0002)
                    drive.CreateStiffnessAttr(600.)
                    drive.CreateDampingAttr(.5)
                    # GPU-compatible ideal unilateral cam. The jaw fixes frame 0
                    # to world orientation; only cam height relative to the jaw
                    # is limited. All five other relative freedoms remain free.
                    # q_jaw >= .0025*sin(open_angle), but wire contact may raise
                    # the jaw above this minimum even with the lever closed.
                    cam = UsdPhysics.Joint.Define(prim.GetStage(),path+'/CamFollower')
                    cam.CreateBody0Rel().SetTargets([prim.GetPath()])
                    cam.CreateBody1Rel().SetTargets([prim.GetParent().GetPath().AppendChild('lever')])
                    cam.CreateLocalPos0Attr(Gf.Vec3f(0.,.0025,.0011))
                    cam.CreateLocalPos1Attr(Gf.Vec3f(0.,-.0025,0.))
                    limit = UsdPhysics.LimitAPI.Apply(cam.GetPrim(),'transZ')
                    limit.CreateLowAttr(-.01)
                    limit.CreateHighAttr(0.)
                    cam.CreateCollisionEnabledAttr(False)
            return prim
        addon = dict(func=spawn,activate_contact_sensors=spec.name in ('clamp','wire','lever'),
                     collision_props=PhysxCollisionPropertiesCfg(contact_offset=.00005, rest_offset=0.),
                     physics_material=sim.RigidBodyMaterialCfg(static_friction=1.2,dynamic_friction=.9))
        if spec.mass is not None:
            addon['rigid_props'] = PhysxRigidBodyPropertiesCfg(solver_position_iteration_count=128, solver_velocity_iteration_count=16, max_depenetration_velocity=5., max_linear_velocity=2., max_angular_velocity=120., sleep_threshold=0., stabilization_threshold=0.)
        return Object(name=spec.name, prim_path='{ENV_REGEX_NS}/'+spec.name,
                      object_type=ObjectType.RIGID if spec.mass is not None else ObjectType.BASE,
                      usd_path=str(HERE/'assets'/f'{spec.name}.usd'),
                      initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos),rotation_xyzw=(0.,0.,0.,1.)),spawn_cfg_addon=addon)

    def build(embodiment, control_hz=None, **kwargs):
        loaded = kwargs["task"]
        if not any(isinstance(obj,task.Model) for obj in loaded.objects):
            return native_build(embodiment,loaded,control_hz)
        for side in ('left_robot','right_robot'):
            robot = getattr(embodiment.scene_config,side)
            robot.spawn = robot.spawn.replace(collision_props=PhysxCollisionPropertiesCfg(contact_offset=.00005,rest_offset=0.),activate_contact_sensors=True)
        from dataclasses import fields
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab_arena.utils.configclass import make_configclass
        original_fields = [(f.name,f.type,getattr(embodiment.scene_config,f.name)) for f in fields(embodiment.scene_config)]
        sensor = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/clamp',filter_prim_paths_expr=['{ENV_REGEX_NS}/lever','{ENV_REGEX_NS}/wire'],update_period=1./(control_hz or 20.),history_length=0,max_contact_data_count_per_prim=128)
        sensor_fields = [('jaw_contact',ContactSensorCfg,sensor)]
        import os
        if os.environ.get('WAGO_CONTACT_DEBUG'):
            body_names = ('base','link1','link2','link3','link4','link5','gripper','tip_left','tip_right')
            fixture_stage = __import__('pxr.Usd',fromlist=['Stage']).Stage.Open(str(HERE/'assets'/'fixture.usd'))
            fixture_shapes = [p.GetName() for p in fixture_stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
            for side,other in [('LeftArm','RightArm'),('RightArm','LeftArm')]:
                filters = ['{ENV_REGEX_NS}/'+other+embodiment.link_paths[n] for n in body_names]
                labels = [other+'/'+n for n in body_names]
                filters += ['{ENV_REGEX_NS}/wire','{ENV_REGEX_NS}/lever']
                labels += ['wire','lever']
                filters += ['{ENV_REGEX_NS}/fixture/'+n for n in fixture_shapes]
                labels += ['fixture/'+n for n in fixture_shapes]
                for link_name in ('link3','link4','link5','gripper','tip_left','tip_right'):
                    name = 'debug_'+side+'_'+link_name
                    cfg = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/'+side+embodiment.link_paths[link_name],
                        filter_prim_paths_expr=filters,update_period=1./(control_hz or 20.),history_length=0,max_contact_data_count_per_prim=512)
                    sensor_fields.append((name,ContactSensorCfg,cfg))
                    debug_sensors[name] = labels
        embodiment.scene_config = make_configclass('TerminalContactSceneCfg',original_fields+sensor_fields)()
        robot = embodiment.scene_config.left_robot
        original = getattr(robot.spawn.func,'__wrapped__',robot.spawn.func)
        link = embodiment.link_paths['gripper']
        @clone
        def with_tool(path,cfg,*args,**kwargs):
            prim = original(path,cfg,*args,**kwargs)
            child = UsdGeom.Xform.Define(prim.GetStage(),path+link+'/LeverTool').GetPrim()
            child.GetReferences().AddReference(str(HERE/'assets'/'tool.usd'))
            from pxr import Usd, PhysxSchema
            for part in Usd.PrimRange(child):
                if part.HasAPI(UsdPhysics.CollisionAPI):
                    collision = PhysxSchema.PhysxCollisionAPI.Apply(part)
                    collision.CreateContactOffsetAttr(.00005)
                    collision.CreateRestOffsetAttr(0.)
            body = prim.GetStage().GetPrimAtPath(path+link)
            PhysxSchema.PhysxRigidBodyAPI.Apply(body).CreateEnableCCDAttr(True)
            return prim
        robot.spawn = robot.spawn.replace(func=with_tool)
        import os
        if os.environ.get('WAGO_CLOSEUP'):
            from dataclasses import fields
            from isaaclab.sensors import CameraCfg
            from isaaclab_arena.utils.cameras import ArenaCameraCfg
            from isaaclab_arena.utils.configclass import make_configclass
            from yam_arena.ik import matrix_to_quat_xyzw
            eye = task.BASE + np.array([.09,-.12,.10])
            look = task.BASE + [0.,-.015,.004]
            if os.environ.get('WAGO_MOUNT_VIEW'):
                from yam_arena import ik
                from yam_arena.constants import READY
                tcp, rotation = ik.tcp_pose(READY,embodiment.scene_config.left_robot.init_state.pos)
                palm = tcp - rotation @ np.asarray(TCP_OFFSET)
                look = palm + rotation @ np.array([0.,0.,.075])
                eye = palm + rotation @ np.array([.24,-.24,.16])
            forward = (look-eye)/np.linalg.norm(look-eye)
            right = np.cross(forward,[0.,0.,1.]); right /= np.linalg.norm(right)
            up = np.cross(right,forward)
            q = matrix_to_quat_xyzw(np.column_stack((right,up,-forward)))
            cam = CameraCfg(prim_path='{ENV_REGEX_NS}/TerminalCamera',height=240,width=320,data_types=['rgb'],
                spawn=sim.PinholeCameraCfg(focal_length=32.,clipping_range=(.001,5.)),
                offset=CameraCfg.OffsetCfg(pos=tuple(float(v) for v in eye),rot=tuple(float(v) for v in q),convention='opengl'),
                renderer_cfg=embodiment.camera_config.top_cam.renderer_cfg)
            existing = [(f.name,f.type,getattr(embodiment.camera_config,f.name)) for f in fields(embodiment.camera_config)]
            embodiment.camera_config = make_configclass('TerminalCameraCfg',existing+[('terminal_cam',CameraCfg,cam)],bases=(ArenaCameraCfg,))()
        task.HISTORY.update(opened=False, withdrawn=False)
        definition = native_build(embodiment,loaded,control_hz)
        native_configure = definition.env_cfg_callback
        def configure(cfg):
            cfg = native_configure(cfg)
            cfg.episode_length_s = 100.
            cfg.sim.dt = .002
            cfg.decimation = round(1./((control_hz or 20.)*cfg.sim.dt))
            cfg.sim.render_interval = cfg.decimation
            cfg.sim.physics.enable_ccd = True
            cfg.sim.physics.friction_offset_threshold = .001
            cfg.sim.physics.friction_correlation_distance = .0005
            return cfg
        definition.env_cfg_callback = configure
        return definition

    from collections import deque
    pose_history = deque()
    last_step = [-1]
    def states(env,names):
        out = native_states(env,names)
        if not {'wire','lever'} <= out.keys(): return out
        raw = env.unwrapped
        step = int(raw.episode_length_buf[0])
        if step < last_step[0]:
            task.HISTORY.update(opened=False,withdrawn=False)
            pose_history.clear()
        new_step = step != last_step[0]
        last_step[0] = step
        wire_root = raw.scene['wire'].data.root_pos_w[0].detach().cpu().numpy().copy()
        wire_root -= raw.scene.env_origins[0].detach().cpu().numpy()
        out['_wire_origin'] = wire_root
        task.FEEDBACK['wire_origin'] = wire_root.copy()
        task.FEEDBACK['wire_axis'] = quat_xyzw_to_matrix(out['wire'].quat_xyzw)[:,1].copy()
        # Measure net motion: contact solver velocities can spike while the
        # constrained wire pose stays stationary. Axial spin is symmetric.
        now = step * float(raw.step_dt)
        if new_step:
            pose_history.append((now, wire_root.copy(), task.FEEDBACK['wire_axis'].copy()))
        while len(pose_history)>1 and now-pose_history[1][0] >= .5:
            pose_history.popleft()
        out['_wire_stable'] = bool(pose_history and now-pose_history[0][0] >= .5-1e-8
            and all(np.linalg.norm(p-wire_root)<.0002
                    and a@task.FEEDBACK['wire_axis']>np.cos(np.deg2rad(1.))
                    for _,p,a in pose_history))
        depth, lateral, angle, seated = task.measurements(out)
        task.FEEDBACK['lever_angle'] = float(angle)
        task.HISTORY['opened'] = bool(task.HISTORY['opened'] or angle > 80)
        task.HISTORY['withdrawn'] = bool(task.HISTORY['withdrawn'] or (task.HISTORY['opened'] and depth < -.001))
        out['_opened'],out['_withdrawn'] = task.HISTORY['opened'],task.HISTORY['withdrawn']
        clear,tcp_positions = True,{}
        for side in ('left_robot','right_robot'):
            robot = raw.scene[side]
            i = robot.body_names.index('gripper')
            p = robot.data.body_pos_w[0,i].detach().cpu().numpy()
            q = robot.data.body_quat_w[0,i].detach().cpu().numpy()
            tcp = p + quat_xyzw_to_matrix(q) @ np.asarray(TCP_OFFSET)
            tcp -= raw.scene.env_origins[0].detach().cpu().numpy()
            tcp_positions[side] = tcp.tolist()
            joint_ids = [robot.joint_names.index(f'joint{j}') for j in range(1,7)]
            actual_q = robot.data.joint_pos[0,joint_ids].detach().cpu().numpy()
            task.FEEDBACK[side] = (tcp.copy(), quat_xyzw_to_matrix(q).copy(), actual_q.copy())
            axis = task.FEEDBACK['wire_axis']
            nearest = wire_root + axis*np.clip((tcp-wire_root)@axis,-.085,.035)
            clear &= np.linalg.norm(tcp-nearest) > .045 and np.linalg.norm(tcp-task.PIVOT) > .045
            if side == 'left_robot':
                tool_head = tcp + quat_xyzw_to_matrix(q) @ np.array([.076,0.,.04])
                clear &= np.linalg.norm(tool_head-task.PIVOT) > .025
        out['_hands_clear'] = bool(clear)
        # Clamp AABB local Z centre is .000925 above its authored root.
        contact_forces = raw.scene.sensors['jaw_contact'].data.force_matrix_w[0].detach().cpu().numpy()
        contact_forces = np.linalg.norm(contact_forces,axis=-1).sum(axis=0).tolist()
        jaw_root = raw.scene['clamp'].data.root_pos_w[0].detach().cpu().numpy().tolist()
        jaw_travel = float(out['clamp'].pos[2] - task.BASE[2] - .004125)
        out['_clamped'] = bool(seated and .0011 < jaw_travel < .0018 and contact_forces[1]>.02)
        indicator = raw.sim.stage.GetPrimAtPath(raw.scene.env_prim_paths[0]+'/fixture/continuity_indicator')
        if indicator.IsValid():
            UsdGeom.Gprim(indicator).GetDisplayColorAttr().Set([(.05,.8,.1) if out['_clamped'] and abs(angle)<5 else (.8,.05,.03)])
        import os,json
        contacts = {}
        for name,labels in debug_sensors.items():
            sensor = raw.scene.sensors[name]
            total = float(sensor.data.net_forces_w[0].norm())
            if total > .1:
                values = np.linalg.norm(sensor.data.force_matrix_w[0].detach().cpu().numpy(),axis=-1).sum(axis=0)
                contacts[name] = dict(total=total,targets={str(labels[i] if i<len(labels) else i):float(v) for i,v in enumerate(values) if v>.1})
        if os.environ.get('WAGO_CLOSEUP') and step % 10 == 0:
            from PIL import Image
            folder = Path(os.environ.get('WAGO_TRACE','/tmp/wago-trace.jsonl')).with_suffix('')
            folder.mkdir(exist_ok=True)
            image = raw.scene.sensors['terminal_cam'].data.output['rgb'][0,...,:3].detach().cpu().numpy()
            Image.fromarray(image).save(folder / f'{step:05d}.png')
        if os.environ.get('WAGO_TRACE'):
            with open(os.environ['WAGO_TRACE'],'a') as log:
                log.write(json.dumps(dict(step=step,depth=depth,lateral=lateral,angle=angle,seated=seated,
                                         wire=wire_root.tolist(),wire_velocity=out['wire'].lin_vel.tolist(),wire_angular_velocity=out['wire'].ang_vel.tolist(),hands_clear=out['_hands_clear'],clamped=out['_clamped'],resting=out['wire'].resting(.005,.3),axis_resting=task.wire_resting(out),jaw_travel=jaw_travel,jaw_root=jaw_root,wire_axis=task.FEEDBACK['wire_axis'].tolist(),contact_forces=contact_forces,tcp=tcp_positions,tracking=task.FEEDBACK.get('tracking',{}),tool_head=tool_head.tolist(),contacts=contacts,history=task.HISTORY,
                                         success=task.success(out)))+'\n')
        return out
    task_object._wago_local = True
    scene.task_object,scene.build_task_environment,scene.object_states = task_object,build,states

def check_assets():
    """Check portability and the full handle sweep against fixed housing cubes."""
    from pxr import Usd, UsdGeom, UsdPhysics
    paths = sorted((HERE/'assets').glob('*.usd'))
    assert len(paths)==5, 'Expected five bundled assets'
    assert sum(p.stat().st_size for p in paths)<25*1024**2
    for path in paths:
        stage = Usd.Stage.Open(str(path))
        assert stage.GetDefaultPrim(), path
        assert not stage.GetRootLayer().GetExternalReferences(), path
    fixture = Usd.Stage.Open(str(HERE/'assets'/'fixture.usd'))
    lever = Usd.Stage.Open(str(HERE/'assets'/'lever.usd'))
    cache = UsdGeom.BBoxCache(0.,['default'])
    def bounds(prim):
        b = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        return np.array(b.GetMin()),np.array(b.GetMax())
    low,high = bounds(lever.GetPrimAtPath('/Model/handle'))
    polygon = np.array([[low[1],low[2]],[high[1],low[2]],
                        [high[1],high[2]],[low[1],high[2]]])
    pivot = np.array([.00325,.005,.0043])
    for angle in np.linspace(0.,100.,201):
        a = math.radians(-angle)
        r = np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
        moving = polygon @ r.T + pivot[1:]
        for prim in fixture.Traverse():
            if not (prim.IsA(UsdGeom.Cube) or prim.IsA(UsdGeom.Mesh)) or not prim.HasAPI(UsdPhysics.CollisionAPI): continue
            lo,hi = bounds(prim)
            if min(high[0]+pivot[0],hi[0])-max(low[0]+pivot[0],lo[0])<=1e-6: continue
            fixed = np.array([[lo[1],lo[2]],[hi[1],lo[2]],[hi[1],hi[2]],[lo[1],hi[2]]])
            axes = [np.array([1.,0.]),np.array([0.,1.]),r[:,0],r[:,1]]
            overlap = all(min((moving@v).max(),(fixed@v).max())-
                          max((moving@v).min(),(fixed@v).min())>1e-6 for v in axes)
            assert not overlap, f'Handle intersects {prim.GetName()} at {angle:g} degrees'
    print('PASS: five self-contained assets; below 25 MB; handle clears housing throughout 0–100 degrees.')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='check existing assets without changing files')
    if parser.parse_args().check: check_assets()
    else: generate()
