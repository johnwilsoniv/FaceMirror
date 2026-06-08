# S2.5 Frame Curator — current UI status (for the design context)

> Status report back to the design context that produced the mockup + handoff.
> Describes what is **actually built and verified** right now, what matches the
> mockup, what differs, and the open design questions. Self-contained — assumes
> no codebase access.

## What it is / where it sits

S2.5 sits between **S2** (Facial Action Coder — labels time-ranges per action)
and **S3** (analysis). Its job: produce a clean set of frames *representative of
each action*, per patient. Cleaning task, single rejection axis.

- **Stack**: PyQt5 + OpenCV, matching S1/S2/S3 (PyInstaller/Inno packaging later).
- **Built and verified** headless (build, toggle, undo, per-range bulk, accept-all,
  save & next, multi-range grouping all pass). Runs as a desktop window.

## Data facts that shaped the build (verified against the real 111-patient set)

- **Multi-range per action is rare**: BS 3%, SS 7%, RE 2%, ES/ET/SE 0%, SO 2%;
  max ranges for any patient×action = 3. The mockup's 2-range BS is a real but
  ~3% case. Range grouping is implemented and correct, but most pages show one
  range.
- **Frames already pool across all ranges** of an action (the handoff's "second
  range silently dropped" concern does not exist in this build).
- **Source frames are 1080×1920 portrait**, face ~35% of frame, **no black bars**.
  The "black/background" problem was thumbnail letterboxing + a dark theme, now
  fixed (see below).
- **No per-frame confidence from the predictor** (auto-window is binary). A proxy
  is derived (see Confidence below).

## Current layout (as built)

**Left sidebar** (batch navigation — requested by the PI, not in the mockup):
- Patient list, color-coded by status (green=done / amber=partial / gray=todo),
  each row `● <id> [case|CTRL] done/total`.
- **Action map**: grid of the patient's actions; current=blue w/ dark border,
  done=green, todo=gray; **bold** = in the analyzed set (BL/BS/SS/RE/ES/ET/SE/SO),
  thin = present but not analyzed. Click to jump.

**Main card** (matches the mockup):
- **Context bar**: "cleaning big smile (BS) · pooled from K ranges" + green
  "autosave/saved ✓" chip.
- **Subtitle**: "K BS ranges in this NN.Ns clip · <patient id>".
- **Clip-overview timeline**: thin full-clip bar, each S2 range drawn as a green
  R1/R2 block at its real time position (custom-painted).
- **Three count cards**: representative / rejected / flagged-to-review. The review
  card turns tan with brown text when N>0 ("N left").
- **Per-range sub-sections**: header `Range 1 · 16.4–18.2s · 7 frames` + per-range
  `keep all ⇧N · reject all ⌥N`, then a horizontal row of frame cells (time-ordered,
  horizontal scroll if many).
- **Frame cell**: the actual **face-cropped thumbnail** (not a schematic box),
  with overlays — kept = green border + ✓, low-confidence = brown dot + amber
  "review · t.ts" caption, rejected = dimmed/gray border + ✕ + "not characteristic"
  caption, plain kept = "t.ts" timestamp caption.
- **Toolbar**: `✓ accept all predictions A` · `↩ undo ⌘Z` · inline hint "click
  toggles · shift-click a range · everything starts on the model's guess" ·
  primary `save & next action ↵`.

**Theme**: flat, warm off-white app bg (#f4f2ec), white card, semantic green for
keep, neutral gray for reject, brown/tan for review, indigo (#5663c0) primary.
Arial. Fusion style.

## Behaviors (as built)

- Every frame **starts on the model's prediction** (auto characteristic-window =
  kept).
- **Single axis**: keep vs not-characteristic. No reason picker (per PI decision).
- **Click** toggles a frame. **Shift-click** applies the new state to that frame's
  whole range. Per-range `keep all`/`reject all`. `accept all` confirms the model
  prediction for the action.
- **Keyboard**: A = accept-all, ⌘Z = undo, ↵ = save & next, ⇧N/⌥N = keep/reject
  range N. 50-level undo.
- **Autosave** on every edit + navigation; writes `s2_5_curation.json` (full
  provenance: kept frames, auto-prediction, human-edit diff, status, timestamps)
  and a flat `s2_5_curated_frames.csv` for the aggregator.
- **save & next** marks the action done and advances to the next action, then the
  next patient at the end.

## Face crop (the thumbnail fix)

Haar-detects the face once per patient (6-frame median, cached), expands with
forehead/brow + chin + cheek margins, then **aspect-matches the crop to the cell**
so the thumbnail fills with no letterbox. Clinically-relevant regions (brows for
AU01/02/04, chin for AU17) are preserved.

## Confidence proxy (drives the review flag)

No predictor confidence exists, so: a frame is "low-confidence" if its **in-set AU
sum** (sum of the AUs the task evokes — e.g. AU12/AU06 for BS) sits within ±18%
of the keep/reject threshold (the auto-window's lower bound). Those get the brown
review dot; the "flagged to review (N left)" counter = low-confidence frames the
human hasn't yet touched/confirmed. Tunable; on some patients it's 0 (their frames
sit clearly above/below threshold).

## Matches the mockup

Context bar, pooled-from-K-ranges, clip timeline w/ R-blocks, three count cards
incl. tan review card, per-range sections w/ bulk actions + hotkey hints, keep/
reject cells, low-confidence review dot, toolbar w/ accept-all/undo/save-next +
inline hints. Single axis.

## Differs from / adds to the mockup

- **Adds a left batch-navigation sidebar** (patient list + action map) — the PI
  explicitly wanted a batch "map" of patients/actions done vs todo. Not in the
  mockup; sits left of the main card.
- **Real face thumbnails** inside cells (mockup cells were schematic f1/f2 boxes).
- Cells show a **timestamp** rather than an "f1/f2" ordinal.

## Not yet built (deferred polish)

- In-grid **video playback** (play the segment for context).
- **PyInstaller/Inno packaging** into a distributable app like S1/S2.
- A "flag this range as mis-bounded" escape hatch back to S2 (handoff listed as
  an open question; currently boundaries are treated as locked).
- Stale MVP widgets `frame_grid.py` / `frame_thumbnail.py` remain in the folder,
  superseded by `frame_cell.py` / `range_section.py` — pending deletion.

## Open design questions back to you

1. **Low-confidence band** (±18% of threshold) yields 0 flags on some patients.
   Is "review flag" meant to always surface *some* frames (e.g. always the N
   nearest the threshold), or correctly show 0 when the model is confident?
2. **Cell timestamp vs ordinal** — keep the real timestamp, or the mockup's
   f1/f2 ordinal, or both?
3. **Sidebar** — acceptable as the batch-nav home, or should patient/action
   navigation live inside the main card (mockup has no sidebar)?
4. **Rejected-cell affordance** — currently dim + ✕ + "not characteristic". The
   image stays visible (dimmed). Prefer fully greyed / removed image instead?
5. **Mis-bounded-range escape hatch** — in or out of S2.5 scope?

*Written from the verified build state. The window runs; this reflects what a
reviewer sees on screen today.*
