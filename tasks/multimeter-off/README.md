# Set a digital multimeter to OFF

Turn the selector counter-clockwise until the small pointer dot aligns with OFF. The
casing is secured to the bench; both YAM arms and the original station cameras are kept.
The meter and its fixed leads/accessories are the only task asset. This is a selector
manipulation task, not an electrical continuity test.

`task.json` is the reviewer-facing setup, scoring rubric and fidelity description.
`my_task.py` defines `SCENE`, `OBJECTS`, `success`, `distance`, `score`, and `demo`.
`multimeter_addon.py` and `multimeter_runtime.py` provide task-local USD/hinge support.
They load in memory after Isaac Sim starts and do not edit or install native environment
files. Preserve the complete `assets/` directory. No environment patch is needed.

## Asset packaging

The submission is below 25 MB: body and dial are flattened binary USD files
with one shared set of 2048 x 2048 PNG textures and the required MDL materials.
Original mesh geometry, collision settings, poses, hinge and policy code are preserved.
The NVIDIA asset license is retained at `assets/LICENSE.txt`.

## Run and score

Use the challenge's existing `yam_arena`, Inspect Robots and Dream-1 environment. These
commands start from the root of this repository, wherever it was cloned:

```bash
source ~/IsaacLab-Arena/yam.env
TASK="$PWD/tasks/multimeter-off/my_task.py"

inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task="$TASK" -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --seed 0 \
  --instruction "Turn the multimeter dial counter-clockwise to OFF" \
  --scorer task_score --max-steps 600 --store-frames \
  --log-dir /tmp/multimeter-off-demo

inspect-robots run --embodiment yam_arena --policy dream \
  -E task="$TASK" -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --seed 0 \
  --instruction "Turn the multimeter dial counter-clockwise to OFF" \
  --scorer task_score --max-steps 2400 --store-frames \
  --log-dir /tmp/multimeter-off-dream
```

Run one Isaac Sim process at a time. Add `-E headless=false` for a desktop window.
Dream uses the existing policy installation and credentials. The task does not create,
read, or commit Robocurve upload tokens.

### Recorded evidence and runtime

- Scripted recording: score **1.0**, native `termination_reason=success`, 352 steps,
  17.6 seconds simulated time, approximately 41.6 seconds wall time on an RTX 5080.
- Existing Dream recording: 1,014 applied actions, 50.7 seconds simulated time,
  approximately 108.9 seconds wall time. It was manually interrupted before the
  2,400-step limit; the original evaluation has no aggregate score. Its endpoint is
  scored by replaying those recorded applied actions, without new Dream calls. This is
  explicitly a post-hoc replay score, not a claim of a completed original evaluation.
- Post-hoc Dream endpoint score: **0.0 rounded**, raw `3.653309948503014e-10`
  (dial displacement approximately `4.42e-8` degrees, numerical residual rather than
  meaningful progress). The replay applied all 1,014 recorded actions using the native
  task scorer; it does not relabel the original interrupted evaluation as completed.
  `evaluation_evidence.json` retains the unrounded result.
- Both recordings use base seed **0**, native derived reset seed **1415158438**,
  `object_jitter=0`, 20 Hz control and 320 x 176 policy frames. The layout is fixed.
- A complete 2,400-step model attempt allows 120 simulated seconds. Wall time depends
  on rendering hardware and model latency; the saved timings above describe the
  actual recordings and do not promise a completed-horizon runtime.

The recorded task revision is `74a9b5db9e7f`. The submission additionally caps partial
credit at 0.95 so that an early stop at OFF cannot bypass the required two-second hold.
The successful recorded demonstration already ended through the native success
termination and therefore still earns 1.0. No visible geometry, grasp waypoints or
physics were changed for that scoring correction.

## Numeric rubric

`a` is signed counter-clockwise rotation from reset. The OFF target is +115 degrees.
The dial must stay within 5 mm of its spindle and within 5 degrees of vertical; otherwise
score is zero. Valid intermediate outcomes receive `0.95 * clamp(a / 115, 0, 1)`.
The native success scorer awards 1.0 only after the angle is within 3 degrees of OFF,
linear speed is below 0.005 m/s, and angular speed is below 0.05 rad/s for two consecutive
seconds. Final-state scoring is used, so undoing progress reduces the score.
`distance()` reports angular error in radians. Pinching is the demonstrated technique;
the outcome rubric does not secretly require a particular grasp strategy.

## Reproduce the demonstrated pinch and run tests

```bash
python tasks/multimeter-off/my_task.py --window
python -m pytest tasks/multimeter-off/test_task.py -q
```

Direct execution runs the same scripted `demo()` with an optional close-up camera and
per-fingertip contact measurements. It verifies opposing contacts above the disk,
not just a correct final angle. The ordinary model policy receives only the original
station camera views. Diagnostic files are saved outside the repository by default.

## Simulation scope and real-world counterpart

See `task.json` for every starting object, size, material and pose. The case is fixed by
an abstract underside mounting fixture; the spindle is ideal. The imposed one-way stop
represents a task fixture/protocol and is not a stock property claimed for all meters.
A physical recreation needs equivalent workholding and the same allowed turn direction.
The real meter should be disconnected from live circuits and not used for electrical
measurement during this selector manipulation trial.

Buttons, display, flexible leads, electrical operation and detents are not functional.
No grasp weld or dial motor is used. Contact friction and collision decomposition are
approximations, not hardware-validated measurements.

## Media upload and pull request

Review videos are H.264 MP4, yuv420p, at 20 fps with `+faststart`, and each is below 50 MB.
The setup image shows only the initial station overview, without a close-up panel. Only simulation
views and neutral labels are used: no terminal, desktop, usernames or team identifiers.
The videos and setup image are uploaded to the task board separately, not committed.

From this repository root, after setting the team's credentials **in the shell only**:

```bash
# Set ROBOCURVE_TEAM and ROBOCURVE_TOKEN in your shell; never save tokens in this folder.
python tools/upload_media.py tasks/multimeter-off \
  --video "$HOME/yam-eval/multimeter-off-review-media/model_run.mp4" \
  --demo "$HOME/yam-eval/multimeter-off-review-media/demo.mp4" \
  --screenshot "$HOME/yam-eval/multimeter-off-review-media/setup.png"
```

The uploader fills the real `media` ID in `task.json`. Until then its empty value is a
submission blocker, not a placeholder media ID to publish. Before opening a public PR,
confirm that the signed-in task board has no conflicting pending claim, that the
recorded scores match the uploaded videos, and that the media ID is populated.

```bash
python -c "import json; d=json.load(open('tasks/multimeter-off/task.json')); assert d.get('media', '').startswith('M-'), 'Upload media first'"
git add tasks/multimeter-off
git commit -m "Add multimeter OFF task"
git push -u origin multimeter-off
gh pr create --repo robocurve/benchmark-tasks --base main --fill
```

If a PR is already awaiting review, follow the repository's waiting-task rule rather
than pushing an unreviewed replacement. Submit only this task folder.
