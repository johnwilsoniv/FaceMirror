# S25 Auto-Curator — method & validation (as deployed)

Reverse-engineered from the curating clinician's hand selections on the 10-patient
diverse ground-truth set (95 status=done patient×action instances, 8121 frames).
All numbers are **leave-one-patient-out cross-validation F0.5** (precision-leaning,
per the "purer phenotype" objective). Honest held-out scores, not fit scores.

## Clinician's principles (the model we are encoding)
1. Keep frames where the REQUESTED action is at/near maximal expression.
2. Skip delayed onset (subject doesn't start immediately).
3. Stop at early relaxation / re-do (don't keep the dip between two attempts).
4. Reject blinks (AU45) — EXCEPT eye-closure tasks (ES/ET/BK) where closure is the target.
NOT encoded: rejecting "contaminant" AUs (e.g. AU17 during BS) — that was a
spurious correlation, not the clinician's reasoning.

## Rule archetypes
- REST (BL): keep quiet frames (low total AU) + no blink.
- EXPRESSION, frac-of-peak + position: keep task-signal >= frac*peak AND
  position >= pos (held portion), minus blinks. Used by RE, ES, ET, SE, SO.
- EXPRESSION, plateau: smooth the task signal, keep the LONGEST contiguous run
  held >= rel*peak (encodes onset-skip + early-stop), minus blinks. Used by BS, SS.
- POSITION-only: no panel AU for the task -> keep position >= pos. Used by PL/BC/LT.

## Key findings
- **Focused AU dyads beat the full in-set sum** for the smile complex (your
  hypothesis). Summing all 7 BS AUs diluted the signal.
    - BS task signal = AU10+AU25  (0.62 -> 0.86)
    - SS task signal = AU12+AU23  (0.70 -> 0.75 grid-selected; ~0.84 achievable)
- **Stronger-side reading** (higher key-AU-peak hemiface) for BK/BS/SS: ~40% of
  patients are right-dominant, so always-left read the weaker side. Applied only
  where CV proved it (NOT RE, which was already near-perfect on default side).
- **The 6 no-in-set actions** (BK/WN/PL/BC/LT/FR) went from keep-all (F0.5 .31-.52)
  to genuinely good (.73-.95) via single defining AU (WN->AU09, BK->AU45) or
  position (PL/BC/LT).

## Deployed CV F0.5 (vs original plateau selector)
| action | orig | final |  rule |
|--------|------|-------|-------|
| BL | .97 | .98 | rest |
| RE | .99 | .98 | frac+pos |
| ES | .91 | .90 | frac+pos (no blink filter; eye task) |
| ET | .91 | .93 | frac+pos (eye task) |
| BS | .62 | .86 | plateau + AU10+AU25 + stronger-side |
| SS | .70 | .75 | plateau + AU12+AU23 + stronger-side |
| SE | .90 | .90 | frac+pos |
| SO | .61 | .69 | frac+pos (n=5) |
| WN | .39 | .90 | AU09 |
| FR | -   | n=1 | AU04 (insufficient data) |
| BK | .31 | .80 | AU45 + stronger-side |
| PL | .52 | .95 | position |
| BC | .51 | .80 | position |
| LT | .49 | .73 | position |

## Honest limitations
- BS/SS/SO/LT (0.69-0.86) are the hardest + smallest-n actions. SS grid is
  sensitive on n=10 (0.75 vs 0.84 across grid variants); we shipped the honest
  grid-selected 0.75 rather than hand-pick the optimum. FR is n=1 (no CV).
- SO/LT dyad candidates (combo search) looked strong but were NOT adopted (n=5,
  overfit risk) — they stay on full-sum until the +30 expansion confirms.
- The right lever for the weak actions is MORE DATA (the +30 patients), not
  cleverer rules on n=10.

## Files
- s25_auto_curator.py     — fit + leave-one-patient-out CV harness
- s25_au_combo_search.py  — dyad/triad search (diagnostic)
- s25_auto_review.py      — human vs new-auto vs old-auto comparison sheets
- s25_auto_params.json    — DEPLOYED per-action rule params (curator reads at startup)
- data_manager.auto_keep_frames() — applies the rules; _legacy_auto_keep() fallback

## Re-fit after curating more patients
Run:  python3 s25_auto_curator.py   (writes /tmp/s25_auto_params.json)
then copy to S3 Data Analysis/s25_auto_params.json. Existing hand-curation is
never modified — new params only change the auto baseline for uncurated actions.
