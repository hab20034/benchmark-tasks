# Refit a fridge door bin on its two hidden supports

**Instruction:** "Put the door bin back on the fridge door at the bottom position, pushed down onto both supports until it stops, then let go."

Refrigerator door bins hang on moulded support bosses on the inside of the door's two side walls (dikes). Pockets
in the bin's end walls, open at the bottom, drop over the bosses. The makers' manuals give the same refit procedure:
"slide [the bin] in above the desired support button and push down until it stops" (LG owner's manual
MFL68680403, p. 24; Whirlpool ET25DQ use and care guide, "Adjusting the door bins"). GE's 22 series manual (p. 10)
says to "position the bin above the rectangular molded supports on the door" and "slide the bin down onto the
support to lock it in place". The bin is 457 mm long, and it fits between the dikes with only 2 mm each side. Both
pockets are hidden behind the dikes while it goes down. So the bin has to come in level and square above the bosses,
and both ends have to go down together. If one end goes first, the bin wedges between the walls. If it is a few
millimetres too far forward, the end walls land on top of the bosses, and the bin sits 25-30 mm high but looks
fitted.

## Running a trial

```
inspect-robots run --embodiment yam_arena -E task=refit_fridge_door_bin_on_its_supports.py \
    --instruction "Put the door bin back on the fridge door at the bottom position, pushed down onto both supports until it stops, then let go." \
    --policy yam_scripted --scorer task_score --max-steps 1200
```

- Scene: `station`. Control runs at 20 Hz. The run limit is 1,200 steps (60 s). The scripted demo takes 21.9 s and reaches
  success at step 382 (19.1 s). A demo trial takes about 2 minutes on one GPU, and a Dream-1 trial about 7 minutes.
- No object jitter. The bin starts in one fixed layout, so the seed changes nothing.
- `--scorer task_score` gives 1.0 on success. Otherwise it gives the score at the end of the run (`SCORE_OVER = "end"`).
- Offline checks: run `pytest test_refit_fridge_door_bin_on_its_supports.py`. These generated scorer tests cover
  the start, goal, partial and cheat layouts.

## Scoring

The scorer reads the bin's pose and velocity and the two boss caps' slide positions. To find each gripper's centre
point for the release check, it reads the robot's joint state.

| Rule | Value |
|---|---|
| Engaged end | The centre of the boss top lies inside that end's pocket. That means it is within the pocket's 22 mm width and at least 5 mm up inside the pocket mouth. |
| Locked end | The end is engaged. Its cap is pushed to within 0.5 mm of its 4 mm stop. The pocket roof rests within 2 mm of the cap top. |
| Level | The bin's up axis is within 3 deg of vertical. A fitted bin hangs on the bosses and leans back about 1 deg onto the liner. |
| Released and still | Each gripper centre is at least 30 mm from the bin's bounding box. Every bin corner moves less than 0.5 mm (plus 0.25 deg of turn) per 50 ms step, and the caps move less than 0.2 mm per step. All of this holds for 1 s. Every tier needs it. Stillness is judged from pose differences because the bin's reported velocities chatter while it rests on contacts. |
| Tiers | 1.0: both ends locked and the bin level. 0.75: both ends engaged. 0.5: one end engaged. 0.25: neither end engaged, but the bin is between the dikes with its back within 25 mm of the liner and 5 mm or more off the bench. |
| Sticky zero | The bin falls (drops more than 45 mm in one 50 ms step, which is 0.9 m/s), turns upside down, or leaves the table. |
| Success | 1.0, held for 1 s (`SUCCESS_HOLD_S`). |

**Measuring it in a real trial:**
1. Before the trial, fit the bin by hand and push it home. Mark its bottom height H at each end on the dike. H is
   15 mm here.
2. After the robot lets go, measure each end with a ruler:
   - pushed home (locked): within 1 mm of H;
   - resting on the snug fit: H + 4 mm;
   - end wall sitting on top of its boss: H + 22 mm or more.
3. To check that an end is engaged, hook a spring scale on its corner and pull 5 N toward the robot. An engaged end doesn't come forward.
4. A fall (a free drop of about 40 mm or more) is read from the phone video.
5. Use a phone video to confirm the release and the 1 s of stillness.

## Layout and dimensions (metres, world frame; arm bases at (0, +/-0.305); table top z = 0.7475)

| Object | Type | Material | Size | Mass | Pose |
|---|---|---|---|---|---|
| Door section | fixed articulation (base plus two slide-jointed caps) | white ABS liner on foam | 545 x 40 x 200 mm door; liner face at x 0.465 | fixed | clamped upright, inside toward the robot |
| Dikes (2) | part of the door, convex prisms | ABS | 12 mm thick, from the liner to x 0.260, 200 mm tall; inner faces at y +/-0.2305 (461 mm apart); 5 mm lead-in on the front inner edge | - | - |
| Support bosses (2) | part of the door, convex prisms | ABS | 20 mm (x) x 12 mm (z) x 6 mm proud; top edges tapered 4 mm (12 mm flat) | - | centred x 0.330, z 0.025-0.037 above the table |
| Snug-fit caps (2) | slide links, 4 mm travel, 4.0 N static / 3.5 N sliding friction | - | 10 x 5 x 3 mm | 2 g | on each boss top; up at the start |
| Door bin | free rigid body, 30 boxes | blue-tinted clear polystyrene, 2.5 mm walls | 457 (y) x 203 (x) x 127 mm; 90 mm front rail with a 12 x 20 mm stiffening rim along its top | 0.45 kg | start: front face x 0.045, bottom on the mat, centred on y 0 |
| Bin pockets (2) | part of the bin | - | open at the bottom, 22 mm wide (x), 25 mm tall, recessed 7 mm; centred 70 mm from the front face, ahead of the centre of mass (about 94 mm) | - | fitted: bin front face x 0.260, bottom z 0.015, leaning back onto the liner |
| Mat | fixed box | rubber | 230 x 464 x 3 mm | fixed | centred (0.140, 0) |

Friction coefficients: polystyrene 0.40/0.35, ABS 0.35/0.30, rubber 0.90/0.80.

## Real-world parts list

- Door bin: an 18 x 8 x 5 in refrigerator fresh-food door bin, for example GE WR71X10957 (which replaces
  WR71X10231, WR71X10722 and WR71X10944) or an equivalent LG or Whirlpool door bin that hangs on support buttons.
- Door section: the lower 200 mm of a fresh-food door liner from a refrigerator model that takes this bin (the bin's
  parts listing names the compatible models), with its side walls and lowest support bosses. Clamp it upright to
  the bench. A scrap door from the same model also works.
- A 3 mm rubber bench mat, a steel ruler, and a phone for video.

## Simplifications

- The door is only its lower 200 mm section.
- These values are estimates, because the makers don't publish them: the front-rail height, wall thickness and
  mass, the pocket and boss sizes, the 2 mm side gap, and the snug-fit force.
- The snug fit over the last 4 mm is modelled as a friction-held cap on each boss top. The pocket roof pushes the
  cap down to its stop, and the cap stays there.
- The bin is rigid.
- The boss tops and dike edges are chamfered instead of rounded.

## Licences

Every object is built from the program's primitive shapes and the task's own procedurally generated convex meshes,
using its own kit modules. No third-party models, meshes or textures are used.
