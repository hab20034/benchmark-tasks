# Fill Monday and Thursday in a snap-lid weekly pill organizer

**Instruction:** "Put one caplet in Monday and one in Thursday of the pill organizer, leaving every lid snapped shut and the other days empty."

A weekly pill organizer is a light plastic strip of seven compartments, one per day, each with its own lid. The lid
is hinged along the back and held shut by a small snap at the front: a tongue hangs from the lid's front edge, and a
hook inside the tongue clicks under a ridge on the compartment's front wall. People open a day by getting a
fingernail under the ledge at the foot of the tongue and lifting until the snap pops, swing the lid up past upright
so it stays open, drop the pill in, then swing the lid down and press its front edge until it clicks. The strip
weighs about 35 g and the snap takes more force than that to release, so lifting a ledge lifts the whole strip
unless the other hand holds it down. Filling the week's organizer is a weekly routine for millions of older adults
and the relatives who look after them.

## Running a trial

```
inspect-robots run --embodiment yam_arena -E task=fill_two_days_of_snap_lid_pill_organizer.py \
    --instruction "Put one caplet in Monday and one in Thursday of the pill organizer, leaving every lid snapped shut and the other days empty." \
    --policy yam_scripted --scorer task_score --max-steps 3000
```

- Scene: `station`. Control at 20 Hz. Run limit: 3,000 steps (150 s). The scripted demo takes 60.6 s (1,212
  steps); a trial takes 20 to 25 minutes on a shared GPU (about 1 s per step).
- No object jitter. Everything starts in one fixed layout, and the seed changes nothing.
- `--scorer task_score` gives 1.0 on success and otherwise the score at the end (`SCORE_OVER = "end"`).
- Offline checks: `pytest test_fill_two_days_of_snap_lid_pill_organizer.py` (generated scorer tests: start, goal,
  partial and cheat layouts).

The scripted demo (`demo()`) is open loop: it uses the start layout and time only. The right hand presses two
spread fingertips (40 mm apart) down on the lids at the Saturday end of the strip while the closed left
fingertips, tipped forward about 50 degrees, get under Monday's ledge, lift it until the snap pops and swing the
lid up past upright. The left hand then picks a caplet
by its ends, lowers it into Monday, pushes the lid back over and presses its front edge until it clicks. The hands
swap for Thursday: the left holds the Sunday end while the right opens, fills and shuts Thursday. The demo patches
`yam_arena.demo.grip_rotation` inside the task module so a waypoint can carry a tilted wrist rotation; the
simulator is unchanged.

## Why the second hand is needed

The snap takes about 1.6 N to release and the strip weighs 0.34 N. In simulation control runs with the same
fingertip motion and no holding hand, the strip slid 2.5 mm away from the robot and the fingertip slipped off the
ledge with the lid still shut; with the strip pinched across its depth at the far end instead of pressed flat, the
strip pivoted up about the pinch (its near end rose 7 mm, tilting 4 deg) and the lid stayed shut. Pressed flat with
two spread fingertips at the far end, the strip stays put and the lid pops open.

## Scoring

The scorer reads the organizer's pose, its seven lid hinges and seven catches, and the caplets' poses. The release
check reads the robot's joint state to find each gripper's centre point.

| Rule | Value |
|---|---|
| Sticky zeros (checked every step) | A caplet's centre ever inside a compartment other than Monday or Thursday; two caplets ever inside one compartment; the organizer ever tipped more than 45 deg; the organizer or a caplet ever more than 20 mm below the table top. |
| Required for any credit, held 1 s | Both gripper centres at least 30 mm from the organizer's bounding box (lids included). Between 50 ms steps the organizer and every caplet move less than 0.5 mm and every lid turns less than 0.25 deg. |
| In a day | A caplet's centre lies inside that compartment: within 17.8 mm of the centre line front to back, within 12.9 mm of the compartment's centre along the strip, and between the floor and the rim (1.2 to 21.5 mm above the organizer's underside). |
| Snapped | The lid's hook centre is at least 0.5 mm below the catch's centre (the hook is under the catch, so the lid cannot open without pushing the catch back). A lid resting shut on top of the catch is not snapped. |
| 1.0 | Exactly one caplet in Monday, exactly one in Thursday, none in any other day, all seven lids snapped, and the organizer flat within 3 deg, within 30 mm of where it lay and within 3 mm of the table. |
| 0.75 | As 1.0, but one lid not snapped. |
| 0.5 | Exactly one caplet each in Monday and Thursday and none elsewhere, but two or more lids not snapped, or the organizer moved more than 30 mm or tilted. |
| 0.25 | Exactly one caplet in only one of Monday or Thursday, that lid snapped, nothing in any other day. |
| 0 | Anything else, including doing nothing, a caplet left on top of a lid, and anything still held or moving. |
| Success | 1.0, held for 1 s (`SUCCESS_HOLD_S`). |

**Measuring it in a real trial:**
1. Outline the organizer's footprint on the table before the trial.
2. After the robot lets go and everything is still, photograph the organizer from above (the lids are
   translucent) and count the caplets in each day.
3. Pick the organizer up and hold it upside down over a tray for 2 s: a snapped lid stays shut; a lid that only
   rested shut drops open.
4. Measure any shift from the outline with a ruler.
5. Judge the sticky zeros (a caplet ever in a wrong day, two caplets ever in one day) from the overhead video.

## Layout (metres, world frame; arm bases at (0, +/-0.305, 0.750); table top 0.7475)

| Object | Type | Material | Size | Mass | Pose |
|---|---|---|---|---|---|
| Pill organizer | free articulation: strip base, 7 hinged lids, 7 sprung catches | translucent polypropylene | 188 x 38 x 23 mm overall (listed 7.4 x 1.5 x 0.9 in) | 0.035 kg (base 17.5 g, lids 2 g each, catches 0.5 g each) | footprint centre (0.190, -0.0134, 0.7475), long axis along y, lids' tongues toward the robot (-x) |
| Caplet A | free rigid stadium | film-coated tablet | 19 x 9 x 7 mm | 0.0013 kg | (0.085, +0.045), lying flat, long axis along x |
| Caplet B | free rigid stadium | film-coated tablet | 19 x 9 x 7 mm | 0.0013 kg | (0.085, -0.045), lying flat, long axis along x |
| Caplet C (spare) | free rigid stadium | film-coated tablet | 19 x 9 x 7 mm | 0.0013 kg | (0.085, 0.000), lying flat, long axis along x |

Organizer detail (organizer frame: origin at the footprint centre on the table, x toward the back, y along the
strip with Sunday at +y, z up):

| Feature | Dimensions |
|---|---|
| Compartments | 7 at a 26.86 mm pitch; inside 25.86 mm wide, 35.6 mm front to back, 20.3 mm deep; floor and walls 1.2 mm, dividers 1.0 mm |
| Day centres (y) | Sun +80.6, Mon +53.7, Tue +26.9, Wed 0, Thu -26.9, Fri -53.7, Sat -80.6 mm |
| Lid | 25.86 x 39 x 1.5 mm, overhangs the front wall by 1 mm; hinge along the back top edge (x = +19, z = 21.5 mm); opens to a 110 deg stop; raised capital 14 x 10 mm, 2 mm strokes |
| Tongue | 12 mm wide, 1.2 mm thick, hanging from the lid underside to 12.5 mm above the table, 1.8 mm in front of the wall |
| Fingernail ledge | at the tongue's foot, 2.5 mm proud of the tongue, 1.2 mm thick, 12 mm wide |
| Hook | rounded rib 2.4 mm diameter, 8 mm long, on the inside of the tongue, standing 1.6 mm in from it; centre 13.9 mm above the table when shut |
| Catch | rounded ridge 3 mm diameter, 8 mm long, centred on the front wall's outer face at 16.5 mm; it can be pushed 2.5 mm back into the wall against a 1.2 N/mm spring with 0.2 N preload |
| Snap | 1.3 mm of hook-over-catch overlap; the hook rides over the catch when the lid is about 4 deg open |

## Forces (estimates, no maker figure found)

- Lifting the ledge to release the snap: about 1.6 N, the peak as the hook rides over the catch (catch spring
  1.2 N/mm with 0.2 N preload, 31 deg initial contact angle, friction about 0.3), plus the lid's 2 g.
- Pressing the lid shut: the rounded snap is symmetric, so about 1.6 N at the front edge, about 2.3 N 10 mm behind
  it (where the demo presses) and about 3.4 N at mid-lid.
- An upside-down organizer puts only the lid's own weight (0.02 N) on the snap, so a snapped lid stays shut.
- To calibrate a real organizer, hook a 0-5 N spring scale under the ledge and lift, then press the lid's front edge
  with a push gauge; replace the organizer if either is outside 1-3 N.

## Real-world parts list

- One extra-large weekly (7-day, one compartment per day) snap-lid pill organizer, such as the Ezy Dose Weekly Pill
  Planner XL (Apothecary Products), translucent polypropylene, listed 7.4 x 1.5 x 0.9 in.
- Three film-coated oblong multivitamin caplets about 19 x 9 x 7 mm (any adult multivitamin of that size; a placebo
  or vitamin is used, no prescription medicine).
- A plain table top (laminate, friction about 0.4).

Safety: no hazards. The objects are light plastic and vitamin caplets; trials run with the arms' usual guarding and
e-stop.

## Licences

Every object is built from the simulator's box and cylinder primitives by this task's own code. No third-party mesh,
scan, texture or font is used. The kit modules `yam_parts.py` and `yam_joints.py` are the program's own unmodified
modules.

## Simplifications

- Rigid primitives throughout. The real snap's flex is in the lid's plastic tongue; here the tongue is rigid and the
  catch ridge is a sprung slide in the front wall, which gives the same release and closing forces and the same
  geometry at the hook.
- Inner sizes, the tongue, ledge, hook and catch, and the 35 g mass are estimates from the listed outer size and the
  maker's photos.
- The day letters are raised white strips (the real ones are printed or moulded).
- Caplets are flat-sided stadiums; real caplets have domed faces.
