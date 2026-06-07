"""
S25 Auto-Curator REVIEW renderer.

For a set of (patient, action) pairs, render a PNG comparison sheet:
  Row 1 = HUMAN kept frames
  Row 2 = NEW auto-curator kept frames
  Row 3 = OLD plateau selector kept frames
Frames the row does NOT keep are shown dimmed, so agreement/disagreement is
visible at a glance. One sheet per (patient, action).

Run:  python3 s25_auto_review.py            # default sample
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

_CURATOR_DIR = ("/Users/johnwilsoniv/Documents/SplitFace Open3/.claude/worktrees/"
                "pilot15-static-mode-audit/S2.5 Frame Curator")
if _CURATOR_DIR not in sys.path:
    sys.path.insert(0, _CURATOR_DIR)
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import config
from data_manager import DataManager
import s25_auto_curator as ac

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path("/Users/johnwilsoniv/Documents/SplitFace/S25 Curated Files/auto_review")
THUMB = 150          # render size per face in the sheet
GAP = 3
LABEL_W = 110


def _font(sz):
    for p in ["/System/Library/Fonts/Helvetica.ttc",
              "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def thumb(dm, pid, frame):
    arr = dm.get_thumbnail(pid, frame)        # RGB ndarray (THUMB_W x THUMB_H)
    if arr is None:
        return Image.new('RGB', (THUMB, THUMB), (40, 40, 40))
    im = Image.fromarray(arr).resize((THUMB, int(THUMB * arr.shape[0] / arr.shape[1])))
    return im


def render_pair(dm, pid, action, params, union_only=True, max_frames=46):
    df = dm.get_frame_df(pid)
    sub = df[df['action'] == action].sort_values('frame')
    if sub.empty:
        return None
    all_frames = sub['frame'].astype(int).tolist()

    st = dm.curation['patients'][pid][action]
    human = set(st.get('kept', []))
    new_auto = set(np.array(all_frames)[ac.predict_keep(sub, action, params)])
    old_auto = set(dm.auto_keep_frames(pid, action))

    if union_only:
        # show only frames at least one method keeps (the decision-relevant ones),
        # so faces are large enough to read; subsample if still too many.
        union = human | new_auto | old_auto
        frames = [f for f in all_frames if f in union]
        if len(frames) > max_frames:
            idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
            frames = [frames[i] for i in idx]
    else:
        frames = all_frames

    rows = [("HUMAN", human), ("NEW auto", new_auto), ("OLD auto", old_auto)]
    th = int(THUMB * config.THUMB_H / config.THUMB_W)
    W = LABEL_W + len(frames) * (THUMB + GAP)
    H = 26 + len(rows) * (th + GAP)
    sheet = Image.new('RGB', (W, H), (244, 242, 236))
    d = ImageDraw.Draw(sheet)
    n_keep = {lbl: len(s & set(frames)) for lbl, s in rows}
    d.text((6, 6), f"{pid}  ·  {action}  ·  {len(frames)} frames   "
                   f"[human {n_keep['HUMAN']} · new {n_keep['NEW auto']} · "
                   f"old {n_keep['OLD auto']}]",
           fill=(40, 40, 40), font=_font(14))

    cache = {f: thumb(dm, pid, f) for f in frames}
    for ri, (lbl, keepset) in enumerate(rows):
        y = 26 + ri * (th + GAP)
        d.text((6, y + th // 2 - 8), lbl, fill=(60, 60, 60), font=_font(13))
        for ci, f in enumerate(frames):
            x = LABEL_W + ci * (THUMB + GAP)
            im = cache[f].copy()
            if f not in keepset:
                # dim the rejects so kept frames pop
                im = Image.blend(im, Image.new('RGB', im.size, (244, 242, 236)), 0.72)
            sheet.paste(im, (x, y))
            # green tick strip atop kept frames
            if f in keepset:
                d.rectangle([x, y, x + THUMB - 1, y + 3], fill=(46, 139, 46))
    return sheet


def main(pairs=None):
    dm = DataManager()
    params = json.loads(Path('/tmp/s25_auto_params.json').read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if pairs is None:
        # default: the actions that changed most + a couple already-good ones,
        # using the first patient that has each done.
        want = ['BK', 'WN', 'PL', 'BC', 'LT', 'BS', 'SS', 'SO', 'ES', 'RE']
        pairs = []
        for action in want:
            for pid, node in dm.curation['patients'].items():
                if (isinstance(node, dict) and action in node
                        and isinstance(node[action], dict)
                        and node[action].get('status') == 'done'):
                    pairs.append((pid, action))
                    break

    written = []
    for pid, action in pairs:
        if action not in params:
            continue
        sheet = render_pair(dm, pid, action, params[action])
        if sheet is None:
            continue
        path = OUT_DIR / f"{action}_{pid}.png"
        sheet.save(path)
        written.append(str(path))
        print("wrote", path.name)
    print(f"\n{len(written)} sheets -> {OUT_DIR}")
    return written


if __name__ == '__main__':
    main()
