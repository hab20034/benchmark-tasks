# Repair a shallow wire connection in a lever terminal

Repair a shallow connection: open the right orange lever with the left arm's
mounted tool, withdraw the blue lead, insert its stripped end fully, close the
lever, and withdraw both hands. A bench fixture holds the housing and
supports the seated lead while the right arm parks before the left closes
the lever.
The task uses the original station and both YAM arms. The opening tool is
permanently wrist-mounted with a visible external orange collar and side
bracket; it is not picked up.
The right arm approaches above the lead as the lever opens. The left
tool remains engaged throughout wire withdrawal, insertion and release.
Both arms then retreat concurrently before the left tool closes the lever.

## Run

Use the existing YAM/Isaac Lab environment. From this task folder:

```bash
source ~/IsaacLab-Arena/yam.env
TASK_DIR="$PWD"  # run from benchmark-tasks/tasks/wago-wire-repair
PYTHONDONTWRITEBYTECODE=1 inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task="$TASK_DIR/my_task.py" \
  -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Hold the orange lever open with the left tool while the right gripper withdraws and fully reinserts the blue wire. Release the wire, close the lever and withdraw both arms." \
  --scorer task_score --max-steps 2400 --store-frames \
  --log-dir ~/yam-eval/wago-scripted
```

For Dream:

```bash
source ~/IsaacLab-Arena/yam.env
TASK_DIR="$PWD"  # run from benchmark-tasks/tasks/wago-wire-repair
PYTHONDONTWRITEBYTECODE=1 inspect-robots run --embodiment yam_arena --policy dream \
  -E task="$TASK_DIR/my_task.py" \
  -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Hold the orange lever open with the left tool while the right gripper withdraws and fully reinserts the blue wire. Release the wire, close the lever and withdraw both arms." \
  --scorer task_score --max-steps 2400 --store-frames \
  --log-dir ~/yam-eval/wago-dream
```

Dream uses its trained 320 × 176 image format and requires the normal model configuration and credentials for the installed
environment. This does not promise policy success.

## Setup and scoring

The layout is fixed (`object_jitter=0`); no random placement or seed sweep is
used. The existing station supplies the table, two YAM arms and cameras.
Custom parts are authored in metres with visual polymer/metal colors and
uncalibrated uniform Coulomb contact materials.

| Part | Size / starting layout |
| --- | --- |
| Fixture | 90 × 190 × 8 mm base, 20 × 25 × 30 mm pedestal, bolted to the bench |
| Two-port housing | 13 × 18.3 × 8.15 mm nominal envelope; fixed at world X=0.25 m, Y=-0.16 m, 35 mm above the native tabletop |
| Active lever | Orange 4.7 × 12 × 1.3 mm handle on a passive X hinge, initially closed |
| Spring jaw | Metal 3.5 × 5 × 1.4 mm contact jaw with passive sliding cam follower |
| Blue lead | Free 120 mm rigid wire; 2.7 mm insulation and 1.4 mm copper diameters; 11 mm stripped end; initially inserted 3 mm |
| Rear rest | Fixed 6 × 4 × 1.2 mm support under the insulated tail |
| Reference port | Fixed black 2.7 mm lead, 11 mm copper end, fixed orange lever; not manipulated |
| Left tool | Wrist-mounted fork with 7 × 5.3 × 0.8 mm blade, 76 mm horizontal reach and 40 mm below TCP; external orange collar and side bracket |
| Indicator | 18 × 20 × 30 mm fixed stand with a 12 × 12 × 3 mm red/green visual marker; not a button or resistance meter |

The numeric score and terminal conditions are specified in `task.json` and
implemented in `my_task.py`. Partial credit sums to at most 0.9; native
`task_score` awards 1.0 only after the complete goal holds for two seconds.
The green indicator shows seating plus closure, not the full task history.

## Files and portability

Copy this whole folder together; the task loads its helper and assets by
paths relative to `my_task.py`.

- `task.json`: submission metadata; the uploader adds its media ID.
- `my_task.py`: object specifications, `success`, `score`, `distance`, `demo`.
- `mechanism.py`: original geometry generator and deferred simulator adapter.
- `assets/*.usd`: all five ready-to-load, self-contained models. No textures,
  internet downloads or native environment edits are needed to load them.

To regenerate geometry, run `python mechanism.py` with USD's `pxr` installed.
Run `python mechanism.py --check` to check bundled files and the full lever
sweep without changing files.
To log measured geometry during a run, set `WAGO_TRACE=/tmp/wago-trace.jsonl`.
Logs and recordings belong outside this folder. Set `WAGO_CLOSEUP=1` alongside
`WAGO_TRACE` to record a separate diagnostic view next to the trace. This view
is not added to the policy observations.

## Fidelity and provenance

This is **original simplified WAGO-style training geometry**, not official
WAGO CAD. External dimensions are based on the 221-412: 13 × 18.3 × 8.15 mm;
stripping length is 11 mm. See [WAGO handling instructions](https://www.wago.com/us/221-handling-instructions)
and [WAGO's CAD product listing](https://wago-embedded.customer-domain.wago.com/3d-cad-models/221-412-splicing-connector-with-levers-for-all-conductor-types-max-4-mm-2-conductor-transparent-housing-surrounding-air-temperature-max-85-c-t85-wago?info=wago%2Fpg07%2Fserie221%2F0221-0412_0999-0962.prj).

The housing is opaque for visibility. The entry has a simplified beveled guide. Cam, spring, friction and tool geometry
are engineering approximations, not measured manufacturer internals. The left
hand carries a fixed custom narrow tool because normal fingertips are too bulky
for a real-scale lever. The blue lead is a 120 mm rigid solid-core segment;
flexible insulation/cable deformation is not simulated. Capsule collision
shapes give the small wire sections analytic radii and rounded contact ends. The other lever and
reference lead are fixed. The indicator reports geometric seating and lever
closure, not electrical resistance. No live electricity is simulated.

Physics runs at 500 Hz, with 20 Hz control in the commands above. Clamp
contact readings are updated at the control rate, rather than at every
physics substep. This scene is computationally demanding and does not
guarantee real-time playback. The commands use 320 × 176 cameras to reduce
rendering cost. Run one simulator instance at a time.

The scripted controller uses measured arm joints to compensate the native
compliant servos’ tracking error. It also uses ground-truth wire and lever
poses for physical arm control. Dream uses its own actions and the same
physical scene and scorer.

The lever is passive. An ideal unilateral cam constraint lifts a spring-driven sliding clamp;
it transmits force between lever and jaw and allows the closed jaw to rest
on a conductor. The cam surfaces themselves are visual. This avoids unstable
submillimetre surface contact; the cam linkage is not manufacturer geometry. A GPU-compatible D6 joint models
this unilateral coupling; it has no motor or angle-driven target.
The wire is free and retained only by collision/friction: no grasp weld, hidden
wire attachment, pose teleport, or policy-dependent physics. Success requires
opening, withdrawing, reinserting, closing with the spring jaw bearing on the conductor, releasing, and hands clear for two
seconds. Stability requires less than 0.2 mm translation and 1° axis change
over the preceding 0.5 seconds. Rotation about the round wire’s own axis
is irrelevant to this goal. A calibrated pull-force
test is not implemented.

Local validation after the concurrent holding and farther-back grasp update:
one deterministic GPU scripted run (`object_jitter=0`,
20 Hz control) completed with native `task_score=1.0` and termination
`success`. The recorded action sequence lasts about 43 seconds. Contact diagnostics
confirmed continuous left-tool/lever contact during withdrawal and insertion,
with both right fingertips gripping the wire and no measured arm-to-arm
contacts above 0.1 N in that interval. Live
simulation is slower than real time; no real-time performance is claimed.
This is a single-run functional check, not a multi-seed benchmark.
The recorded concurrent scripted trial reached 1.0 after 864 control steps
(43.2 simulated seconds), taking 443.7 seconds wall time on an RTX 5080
with diagnostic contacts and an extra close-up camera. Runtime depends on
camera settings and other running simulations. Budget approximately 15–30 minutes
for a full 2,400-step Dream trial, plus model latency; this is an estimate,
not a real-time guarantee.
A 96-step Dream smoke test completed without errors, with two real model
requests after the 32-step preroll (64 model-controlled steps). Its score
was 0.0 at this short limit; Dream task completion has not been established.

Original code and geometry in this folder are provided under the MIT license.
WAGO is a third-party trademark; no endorsement is implied.
