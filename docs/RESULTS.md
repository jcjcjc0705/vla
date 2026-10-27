# Held-out evaluation: ACT vs. GR00T N1.7

One dataset — 486 episodes / 100,096 frames / 3 cameras — four trained policies, evaluated on cube
poses that appear zero times in the training data.

> **All four models were re-measured on 2026-08-13**, 60 episodes each, under one evaluator. Two of the
> previously published numbers were wrong. The old numbers are kept in the table below, and the reason
> they were wrong is the most useful thing in this document.

---

## Best result per run

Each row is the best-scoring checkpoint from that training run, re-measured over 60 held-out episodes.

| Policy | Best checkpoint | Held-out success | 95% CI | Previously reported |
|---|---|---|---|---|
| **ACT** | step 5 000 | **92 %** (55/60) | [85 %, 99 %] | 85 % |
| GR00T v1 | step 40 000 | 60 % (36/60) | [48 %, 72 %] | 43 % |
| GR00T v2 | step 6 000 | 58 % (35/60) | [46 %, 71 %] | **12 %** |
| GR00T v3 | step 14 000 | 35 % (21/60) | [23 %, 47 %] | 45 % |

```
v1 vs v2    z = +0.19    no difference      <- tune_visual does not affect the score
v2 vs v3    z = +2.56    significant
ACT vs v1   z = +4.05    significant
```

A 52 M-parameter ACT scores 1.5× what a 3.14 B-parameter GR00T scores.

---

## Why the old numbers were wrong

`eval.py` used to send **all five cameras** from `ros.camera_topics` to the policy, while all four models
were trained on three (`front_left` / `front_right` / `wrist`). GR00T's processor selects image keys
like this:

```python
def _ordered_image_keys(self, obs):
    available = {key for key in obs if key.startswith(OBS_IMAGES)}
    if not self.video_modality_keys:        # None for an N1.7 checkpoint
        return sorted(available)            # <- takes whatever the obs dict contains
```

So the third view changed identity from `wrist` to `rear_left`, and two extra views were stacked into
the video tensor:

```
trained on    front_left, front_right, wrist                            V=3
old eval      front_left, front_right, rear_left, rear_right, wrist     V=5
```

It never raised an error. It just scored lower. Timeline:

```
2026-08-11 21:19   v1 / v2 / ACT evaluated     <- broken evaluator
2026-08-12 09:25   eval.py fixed, commit 13d61c6
2026-08-12 10:57   v3 evaluated                <- after the fix
```

**Two judgment failures, both mine:**

1. **Ruling out the bug by testing one model.** After the fix I spot-checked v1 step 40000 and got
   8/20 = 40 %, consistent with the old 43 %, and wrote down that the existing numbers had been
   "suspected and cleared". But 8/20 has a 95 % CI of **[21 %, 61 %]** — it has no power to separate
   43 % from the 60 % measured later. I used it as a disproof anyway.
2. **Skipping the sample that mattered most.** `tune_visual` is precisely the variable that determines
   how badly a model is hurt: a frozen vision tower (v1) treats two extra views as noise, while a
   fine-tuned one (v2) receives out-of-distribution input and collapses. v2 was the most important
   model to re-check, and it was the one I skipped.

⚠️ ACT is **immune** to this bug — `modeling_act.py` builds its input as
`[batch[key] for key in self.config.image_features]`, taking only the cameras the checkpoint declares
and ignoring extras. Its 85 % → 92 % movement is within noise.

---

## What is being measured

**Success** = the cube is lifted 0.05 m, remains within 0.06 m of the gripper, and both conditions hold
for 15 consecutive control cycles. The distance condition rejects a cube that was knocked away; the
15-cycle hold rejects a momentary bounce. Both are necessary.

**Everything is measured in the held-out region.** The cube spawns at θ ∈ [−50°, +50°]; the two bands
`[−35°, −20°]` and `[+20°, +35°]` contain **0 of the 486 training episodes** (verified by counting the
full θ distribution).

The held-out bands sit **inside the range rather than at its edges**, and there are two of them: a
policy that ignores the image and replays an average trajectory collapses in the middle, whereas the
outer edge is also the farthest and hardest reach — using it as the held-out region would confound
"generalizes poorly" with "is simply difficult".

⚠️ **Evaluation is not deterministic.** The same checkpoint, the same seed and the same cube pose
produce different outcomes between runs (ACT step 5000 scored both 10/10 and 9/10) — Isaac's physics
and rendering shift frame arrival timing with GPU load. Every checkpoint therefore gets **3 rounds ×
20 episodes**, with a different seed per round.

---

## The four training runs

The three GR00T runs are causally linked — each was designed to explain the previous one's failure.
⚠️ **And half of what they were explaining was measurement error**, so v3 was built to fix a problem
that did not exist.

⚠️ In the per-checkpoint scans below, **v1 and v2 were measured with the broken evaluator** (five
cameras). Their values are systematically low and are kept only to show the *shape* of the curve —
which step was best — not as absolute numbers. **v3 was measured after the fix, and ACT is immune to
the bug**, so those two are absolute. The trustworthy basis for comparison is the 2026-08-13 table at
the top.

### ACT (baseline)

`lerobot_3cam` 240×320 · batch 64 · lr 1e-5 · ResNet18 **trained** (`optimizer_lr_backbone` 1e-5) · 52 M

| step | per round | total | |
|---|---|---|---|
| **5000** | 19/20 20/20 15/20 · 16/20 17/20 15/20 | **102/120** | **85 %** |
| 10000 | 8/20 12/20 10/20 | 30/60 | 50 % |
| 15000 | 8/20 10/20 4/20 | 22/60 | 37 % |
| 20000 | 15/20 17/20 14/20 | 46/60 | 77 % |

It peaks at 3.2 epochs and then degrades. Training was stopped manually at step 20 800. The curve is
not a clean monotonic decline (90 → 50 → 37 → 77): the quality of any individual checkpoint carries
real variance.

### GR00T v1 (default settings)

`lerobot_3cam` 240×320 · batch 8 · steps 60 000 · lr 1e-4 · vision backbone **frozen** · 1.62 B of
3.14 B trainable

| step | run 1 | run 2 | total | |
|---|---|---|---|---|
| 10000 | 3/60 | 2/60 | 5/120 | 4 % |
| 20000 | 10/60 | 20/60 | 30/120 | 25 % |
| 30000 | 11/60 | 12/60 | 23/120 | 19 % |
| **40000** | 26/60 | 25/60 | **51/120** | **43 %** |
| 50000 | 15/60 | 15/60 | 30/120 | 25 % |
| 60000 | 22/60 | 21/60 | 43/120 | 36 % |

🛑 What was written at the time: "two independent measurements agree closely (43 % and 42 %), so the
number is trustworthy". **That reasoning has a hole** — both runs used the same broken evaluator, so
the agreement proves the bug is deterministic, not that the number is right. **Reproducible is not the
same as correct.** The re-measurement gives 60 %.

Incidentally the peak lands at the same amount of data as ACT's: `40000 × 8 = 320,000` samples, versus
ACT's `5000 × 64 = 320,000` — both 3.2 epochs.

### GR00T v2 (vision unfrozen)

Same data as v1 · batch 24 · steps 16 000 · lr 1e-4 (**unchanged**) · vision backbone **unfrozen**
← the only change · 2.03 B trainable

| step | per round | total | |
|---|---|---|---|
| 2000 | 1/20 0/20 3/20 | 4/60 | 7 % |
| 4000 | 0/20 0/20 0/20 | 0/60 | 0 % |
| **6000** | 0/20 3/20 4/20 | **7/60** | **12 %** |
| 8000 | 0/20 2/20 1/20 | 3/60 | 5 % |
| 10000 | 0/20 1/20 0/20 | 1/60 | 2 % |
| 12000 | 1/20 2/20 2/20 | 5/60 | 8 % |
| 14000 | 1/20 0/20 0/20 | 1/60 | 2 % |
| 16000 | 0/20 0/20 0/20 | 0/60 | 0 % |

🛑 **The entire conclusion drawn from this table was wrong.** Re-measured, v2 scores **58 % (35/60)**,
indistinguishable from v1's 60 % (z = 0.19). Every 0 %–12 % figure above is five-camera measurement
error — v2's fine-tuned vision tower is especially sensitive to an out-of-distribution combination of
views, so the same bug hurt it far more than it hurt the frozen-vision v1.

The diagnosis originally written here was:

> **Catastrophic forgetting.** Training loss is lower than v1's (0.0049 vs 0.0082) while held-out
> performance collapses to a third — better fit, worse generalization. Mechanism: `GrootConfig` has no
> backbone-specific learning-rate field, so unfreezing that 400 M vision tower makes it take 1e-4
> alongside the action head.

⚠️ **The forgetting never happened.** Lower training loss *and* equal held-out performance — it simply
trained better. `GrootConfig` does indeed lack a backbone-specific lr (that fact is unchanged), but it
caused no problem here.

**The lesson worth keeping: a mechanism can be entirely plausible and still not substitute for a
measurement.** That explanation reads as completely sound, and it even predicts the classic signature
of lower loss with worse generalization — for a phenomenon that did not exist.

### GR00T v3 (fixing the lr and the aspect ratio)

`lerobot_3cam_sq` **320×320 letterboxed** · batch 24 · steps 16 000 (as v2) · lr **1e-5** · vision
unfrozen

| step | per round | total | |
|---|---|---|---|
| 2000 | 0/20 0/20 2/20 | 2/60 | 3 % |
| 4000 | 2/20 1/20 4/20 | 7/60 | 12 % |
| 6000 | 6/20 4/20 1/20 | 11/60 | 18 % |
| 8000 | 2/20 0/20 1/20 | 3/60 | 5 % |
| 10000 | 7/20 8/20 4/20 | 19/60 | 32 % |
| 12000 | 3/20 1/20 2/20 | 6/60 | 10 % |
| **14000** | 7/20 10/20 10/20 | **27/60** | **45 %** |
| 16000 | 4/20 5/20 1/20 | 10/60 | 17 % |

This scan was taken with the **fixed** evaluator (2026-08-12 10:57, after the 09:25 fix), making it the
only uncontaminated scan of the four. Step 14000 re-measured on 2026-08-13 gives **35 % (21/60)**,
slightly below the 45 % measured then, within noise.

🛑 **But the motivation for this run was wrong.** It was designed to fix "v2's catastrophic
forgetting", which does not exist. The result is that v3 is the **worst** of the three GR00T runs:

```
          images          lr      tune_visual    re-measured 2026-08-13
v1      240×320          1e-4        off              60 %
v2      240×320          1e-4        on               58 %
v3   320×320 letterbox   1e-5        on               35 %    <- worst, z = 2.56
```

⚠️ v2 → v3 changed **two** variables (lr and letterboxing), so the difference cannot be attributed to
either alone. What can be said:

- **`tune_visual` is not the cause** — v1 vs v2 isolates that variable and shows no difference
- **`lr 1e-5` was adopted to solve a problem that did not exist**
- **Letterboxing fixes a real geometric defect** (next section) and should not be reverted

**The untested cell is therefore "letterbox + lr 1e-4".** That is the next experiment worth running —
not continuing toward even smaller learning rates.

---

## Aspect ratio: a handicap GR00T had and ACT did not

Found by watching rollout video. Both GR00T versions had the **motion structure entirely right** —
turning, descending to the correct height, closing the gripper at the right moment, lifting
immediately — and were only imprecise about the cube's position and angle, as though they knew roughly
where it was.

Tracing it down: GR00T's image preprocessing uses `letter_box_transform=False` and
`crop_fraction=null`, so 320×240 is **stretched directly** to 256×256:

```
horizontal  320 px  ×0.800   cube 8 -> 6.4 px
vertical    240 px  ×1.067   cube 8 -> 8.5 px
```

At a 90° FOV and 0.50 m from the workspace, the scale is 0.32 px/mm, so a 25 mm cube is **8×8 px square**
in the original frame and becomes **33 % taller than wide** after stretching. A rotated square under
anisotropic scaling becomes a parallelogram, and the distortion varies with yaw rather than being a
constant offset, so the model cannot learn it away. **ACT consumes the native 320×240 and never
performs this step.**

The fix: `convert.py --pad-square` letterboxes to 320×320, so GR00T's subsequent resize to 256×256 is
isotropic. Written to a new dataset rather than overwriting (ACT's score was measured on 240×320).

⚠️ **The geometric analysis is sound, but "did the fix help?" remains unanswered** — v3 changed the lr
at the same time, and it scores below the un-letterboxed v1/v2. Whether padding helps or is irrelevant
requires running the "letterbox + lr 1e-4" cell.

---

## Three bugs

Two of my own, one upstream. None of them raised an error.

### The dataloader decoded 300 PNGs per sample and threw them away

ACT's `chunk_size=100` requires 100 steps of future actions per sample. LeRobot fetches them with
`hf_dataset[key][indices]`, and `datasets` formats those 100 rows by **decoding every column**,
including the three PNG image columns. cProfile: 99 % of time in `_query_hf_dataset`, 3030 image
decodes per 10 samples.

Anyone using `use_videos=True` never hits this — the images live in mp4 files, not in the Arrow table.
This project deliberately chose PNG-in-parquet to move video decoding out of the dataloader, and that
is exactly the trigger condition. The fix is to project the columns away before fetching rows.

| | updt_s | data_s | smp/s | 60 k steps |
|---|---|---|---|---|
| `lerobot-train` | 0.441 | 9.575 | 6 | ~7 days |
| **patched** | 0.172 | 0.012 | 348 | **~3 hours** |

⚠️ The patch alone is not enough: LeRobot's `dataloader_multiprocessing_context` defaults to **spawn**,
so workers re-import and never see the patch. On first verification `ds[i]` was 49× faster in the
parent while `data_s` did not move at all — none of the real work happens in the parent.

### Hard-coded evaluation image size wrote off an entire run

`eval.py` resized frames to the task yaml's `record_resolution` (320×240), but v3 was trained on
letterboxed 320×320. Nothing crashed — the scale and position were simply wrong, and **eight
checkpoints × three seeds scored 0/480**, which looks exactly like "the model is bad". Fixed by reading
the expected shape from `policy.config.input_features`.

### Five cameras sent to a three-camera model

The client subscribes to all five `ros.camera_topics`; the dataset contains only the three selected by
`--cameras`. With `video_modality_keys=None`, GR00T's `_ordered_image_keys` returns
`sorted(all observation.images.*)` and stacks all five into the video tensor. Harmless-looking before
letterboxing (same size, and `sorted` keeps the two front cameras in the first two positions); after
letterboxing it raises a `np.stack` ValueError.

🛑 **This is the most consequential of the three, and I judged it harmless at the time.** What was
originally written:

> ⚠️ Suspected that this invalidated the existing numbers; **ruled out** by measurement: v1 step 40000
> re-measured with the fixed evaluator gives 8/20 = 40 %, consistent with the earlier 43 %.

That disproof does not hold: **8/20 has a 95 % CI of [21 %, 61 %]**, which covers the 60 % measured
later — it cannot distinguish the two. And the model spot-checked was the **least affected** one: a
frozen vision tower (v1) merely receives two extra channels of noise, while a fine-tuned one (v2)
receives out-of-distribution input, and the same bug made it look like 12 % instead of 58 %.

**Two lessons**: having set a standard of 60 episodes, do not verify with 20; and to disprove "this bug
affects the numbers", test the sample **most likely to be affected**, not the most convenient one.

---

## Why adjacent checkpoints oscillate: each covers a different patch

Three checkpoints from the same v3 run, on **exactly the same 20 episodes**:

```
step  6000   4/20   succeeded on {3, 5, 10, 16}
step 10000   5/20   succeeded on {7, 8, 14, 17, 20}     <- intersection with the above = 0
step 14000   6/20   succeeded on {4, 6, 8, 10, 16, 20}
```

**6000 and 10000 do not overlap at all.** Measured with Cohen's kappa (agreement after discounting the
base rate):

| comparison | agreement | kappa | |
|---|---|---|---|
| v3 6000 vs 10000 | 55 % | **−0.29** | less consistent than chance |
| v3 10000 vs 14000 | 65 % | 0.13 | almost unrelated |
| v3 6000 vs 14000 | 70 % | 0.21 | moderate |
| **ACT 5000 vs 20000** | 75 % | **0.38** | moderate |

If a model had learned a general "see cube → align" mapping, **hard episodes would be hard for every
checkpoint** and the success sets would overlap heavily. v3 does not behave that way — it trades one
workable region for another every two thousand steps.

ACT looks different: its failures also overlap (`{1, 2, 14}` defeat both checkpoints), so **the
difficulty structure is shared** — which is what "learned the task, limited precision" should look
like.

### This changes what two numbers mean

**43 % is not "a four-in-ten chance on every episode"** — it is "40 % of episodes almost always
succeed, 60 % almost always fail", with the dividing line falling in a different place for every
checkpoint.

**A 5 % ↔ 45 % swing between adjacent checkpoints is not noise** — the checkpoints cover different,
nearly disjoint subregions, and the score depends on how much of the test set this particular patch
happens to cover. Essentially a lottery.

⚠️ This analysis also **overturns the earlier claim that "v3 systematically fails on the right half"**.
Only step 14000 leans left (6:0); step 10000 leans slightly right (2:3) and step 6000 is 3:1. The left
bias of 14000 is an accident of that checkpoint, not a consequence of pad-square or tune_visual —
generalizing from a single checkpoint's pattern to the whole pipeline was too fast.

⚠️ **Every kappa figure here comes from v3**, which is now known to be the worst of the three
configurations. So patchwork behaviour may be **a property of v3's settings** rather than of GR00T in
general — v1 and v2 have never had the same per-episode comparison done. **Concluding that "GR00T
learns region-memory" requires redoing the kappa analysis on the 60 % v1 or v2.** Not yet done.

### Mechanism (inference, evidence weakened)

The original reasoning: **486 episodes cannot support 2.03 B trainable parameters**, so the model has
the capacity to memorize training trajectories by region rather than abstracting a general "cube
position → joint angle" relation; ACT, with only 52 M, has no capacity for region-memory and is forced
to learn the generalizing mapping.

⚠️ **But v2 also has 2.03 B trainable parameters, and it scores 58 %.** The capacity hypothesis cannot
explain why the same capacity did not degrade into patchwork in v2. What actually co-occurs with
patchwork is v3's configuration (lr 1e-5 + letterboxing), not the parameter count.

**Testable**: redo the per-episode kappa on v1 (1.62 B) and v2 (2.03 B) checkpoints. If both land near
ACT's 0.38, patchwork is specific to v3; if v2 is also negative, the capacity hypothesis regains
standing.

---

## Conclusions

- **`tune_visual` has no effect on the score.** v1 60 % vs v2 58 %, z = 0.19, and those two runs differ
  in that one variable alone.
- **`lr 1e-5` was the wrong direction.** It was adopted to block catastrophic forgetting, which turned
  out to be measurement error. The only run using it is the worst of the three (35 %, z = 2.56).
- **ACT leads by a wide margin: 92 % vs 60 %** (z = 4.05). 52 M beats 3.14 B.
- **It is not a speed problem.** The concern that a 3 B model could not keep up with a 30 Hz control
  loop — which would make the comparison unfair — was measured: `loop_overruns = 0`. Ruled out.
- **The most likely explanation is the nature of the task, not the hyperparameters.** One object and
  486 episodes of a single instruction means the language dimension is a constant;
  `modeling_act.py` contains no language processing at all, and it scores 92 %.

**The next step is not further GR00T hyperparameter tuning — it is changing the task**: multiple
objects on the table, with instructions that genuinely carry information. ACT is then *structurally*
incapable, capped at a random 33 %. That work is built but unfinished (see
[Status](../README.md#status)): `data/lerobot_3obj`, 1 868 episodes.

⚠️ And that run uses **256×256 letterbox + lr 1e-5 + tune_visual** — v3's configuration, the worst of
the three on the single-object task. **"letterbox + lr 1e-4" remains the untested cell** and should be
filled in.

---

## Checkpoint retention

Only the best checkpoint from each run was kept, renamed to `best` (releasing 496 GB). The step number
is still recorded in `checkpoints/best/training_state/training_step.json`:

| run | best step | size |
|---|---|---|
| `act_v1` | 5000 | 591 MB |
| `groot_v1` | 40000 | 24 GB |
| `groot_v2` | 6000 | 27 GB |
| `groot_v3` | 14000 | 27 GB |

⚠️ The `last` symlink was removed, so these runs can no longer be `--resume`d. Use
`--checkpoint outputs/<run>/checkpoints/best/pretrained_model`.

---

## Reproducing these four measurements

The three-object task replaced `task/pick_cube.task.yaml` and the scene. To re-test the M5/M6 models,
**both the scene and the task spec must be switched back**:

```bash
# Open assets/pick_cube_1obj.usd in Isaac and press Play. Do NOT cp it over
# pick_cube.usd -- that destroys the current three-object scene.
# task/pick_cube_1obj.task.yaml already points at the backup.
python3 ml/eval.py --task task/pick_cube_1obj.task.yaml \
  --checkpoint outputs/groot_v1/checkpoints/best/pretrained_model --episodes 20 --holdout
```

⚠️ Switching the USD is not sufficient — the camera list (five vs. three) and the resolution
(320×240 vs. 448×336) are read from the task yaml, not from the scene. Image size is handled
automatically: `eval.py` reads it from the checkpoint's `input_features`.

Sweeping all four at once (`eval_sweep.sh` takes `TASK` from the environment):

```bash
for r in act_v1 groot_v1 groot_v2 groot_v3; do
  echo "########## $r ##########"
  TASK=task/pick_cube_1obj.task.yaml EPISODES=20 SEEDS="0 1 2" \
    bash ml/eval_sweep.sh outputs/$r
done 2>&1 | tee outputs/eval_1obj_recheck.txt
```

`task/pick_cube_1obj.task.yaml` is frozen; editing it breaks correspondence with these four models.

---

## Claims this document has had to withdraw

Kept as a record, because each one is the same class of error: **closing an investigation on
insufficient evidence.**

| What was claimed | Why it was wrong |
|---|---|
| v2 unfroze vision → catastrophic forgetting, 12 % | Five-camera measurement error. Actually 58 %, no different from v1 |
| The five-camera bug does not affect the existing numbers | Verified with 20 episodes, on the least-affected model |
| v3 systematically fails on the right half of the workspace | Only step 14000 leans left; an accident of one checkpoint |
| GR00T plateaus at 43–45 % | Both figures were low; actually 58–60 % |
| GR00T learns region-memory (patchwork) | Kappa measured only on v3, which is the worst configuration |

---

## Environment

Isaac Sim 5.1.0-rc.19 · LeRobot 0.6.1 · torch 2.11.0+cu128 · numpy 2.2.6 · NVIDIA RTX PRO 6000
Blackwell (96 GB)

Raw logs: `outputs/eval_sweep_result.txt` (ACT + v1), `eval_sweep_v2.txt` (v2, plus v1/ACT re-checks),
`eval_sweep_v3.txt` (v3).

The original Chinese version of this document, which is the primary record, is at
[zh/EVAL_RESULTS.md](zh/EVAL_RESULTS.md).
