"""
S2.5 Frame Curator — data manager.

Loads the patient roster, per-frame AU+action data, the algorithm's auto-window,
source-video frames (thumbnails), and persists curation state.
"""
import json
import os
import hashlib
import threading
from datetime import datetime, timezone
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

import config


class DataManager:
    def __init__(self):
        self.controls = set(p.stem for p in config.CONTROL_DIR.glob('IMG_*.MOV'))
        self.patients = self._discover_patients()
        self.auto_window = self._load_auto_window()
        self.inset = self._load_inset()
        self.auto_params = self._load_auto_params()   # validated per-action rules
        self.curation = self._load_curation()
        # Thumbnail cache: (pid, frame) -> QImage-ready RGB ndarray
        # One lock guards all three caches below; the prefetch QThread and the
        # UI thread both touch them. Only the (fast) dict ops are locked — the
        # expensive cv2 decode happens OUTSIDE the lock, so prefetch still wins.
        self._cache_lock = threading.Lock()
        self._thumb_cache = OrderedDict()
        self._thumb_cache_cap = 1200   # > frames/patient so a full patient fits
        # Per-frame AU cache: pid -> DataFrame (left side, has action + AUs)
        self._frame_cache = OrderedDict()
        self._frame_cache_cap = 8
        # Face-crop boxes: pid -> (x, y, w, h) in source-video pixels
        self._crop_boxes = {}
        self._cascade = None

    # ---------- discovery ----------

    def _discover_patients(self):
        pids = set(
            c.name.replace('_left_mirrored_coded.csv', '')
            for c in config.PER_FRAME_DIR.glob('*_left_mirrored_coded.csv'))
        pids |= set(
            c.name.replace('_right_mirrored_coded.csv', '')
            for c in config.PER_FRAME_DIR.glob('*_right_mirrored_coded.csv'))
        # Distribution: load EVERY patient found in the chosen data folder (the
        # sidebar is the per-patient selector). Cases first (sorted), controls last.
        pids = sorted(pids)
        cases = [p for p in pids if p not in self.controls]
        ctrls = [p for p in pids if p in self.controls]
        return cases + ctrls

    def is_control(self, pid):
        return pid in self.controls

    def _load_auto_window(self):
        """Returns {(pid, action): [frame, ...]}."""
        out = {}
        if not config.AUTO_WINDOW_CSV.exists():
            return out
        df = pd.read_csv(config.AUTO_WINDOW_CSV)
        for _, r in df.iterrows():
            wf = r.get('window_frames')
            frames = []
            if isinstance(wf, str) and wf.strip():
                frames = [int(x) for x in wf.split(',')]
            out[(str(r['patient_id']), str(r['task']))] = frames
        return out

    def _load_inset(self):
        """Returns {action: [au, ...]} of in-set AUs for label display."""
        out = {}
        if not config.INSET_CSV.exists():
            return out
        df = pd.read_csv(config.INSET_CSV)
        for task in df['task'].unique():
            out[str(task)] = df[(df['task'] == task) & (df['in_set'])]['au'].tolist()
        return out

    # ---------- per-frame AU data ----------

    def get_frame_df(self, pid):
        """Per-frame DataFrame (left side) with action + AU columns."""
        with self._cache_lock:
            if pid in self._frame_cache:
                self._frame_cache.move_to_end(pid)
                return self._frame_cache[pid]
        csv = config.PER_FRAME_DIR / f'{pid}_left_mirrored_coded.csv'
        if not csv.exists():
            csv = config.PER_FRAME_DIR / f'{pid}_right_mirrored_coded.csv'
        if not csv.exists():
            return None
        df = pd.read_csv(csv)
        df['action'] = df['action'].astype(str).str.strip()
        with self._cache_lock:
            self._frame_cache[pid] = df
            if len(self._frame_cache) > self._frame_cache_cap:
                self._frame_cache.popitem(last=False)
        return df

    def actions_for_patient(self, pid):
        """Ordered list of actions present for this patient (canonical order)."""
        df = self.get_frame_df(pid)
        if df is None:
            return []
        present = set(df['action'].unique())
        return [a for a in config.CANONICAL_ACTIONS if a in present]

    def frames_for_action(self, pid, action):
        """Sorted frame indices coded as this action (pooled across all ranges)."""
        df = self.get_frame_df(pid)
        if df is None:
            return []
        return sorted(int(f) for f in df[df['action'] == action]['frame'].tolist())

    def get_ranges(self, pid, action, gap=3):
        """Reconstruct the distinct S2 ranges for an action from frame-index gaps.
        Returns [{idx, start_f, end_f, start_t, end_t, frames:[...]}], time-ordered.
        Multi-range is rare (~3-7% of smiles) but provenance is preserved."""
        df = self.get_frame_df(pid)
        if df is None:
            return []
        sub = df[df['action'] == action].sort_values('frame')
        if len(sub) == 0:
            return []
        frames = sub['frame'].astype(int).tolist()
        ts = sub['timestamp'].astype(float).tolist() if 'timestamp' in sub.columns \
            else [f / 30.0 for f in frames]
        f2t = dict(zip(frames, ts))
        ranges = []
        start = prev = frames[0]
        run = [frames[0]]
        for f in frames[1:]:
            if f - prev > gap:
                ranges.append((start, prev, run))
                start = f
                run = []
            run.append(f)
            prev = f
        ranges.append((start, prev, run))
        out = []
        for i, (sf, ef, run) in enumerate(ranges):
            out.append({
                'idx': i + 1,
                'start_f': sf, 'end_f': ef,
                'start_t': f2t.get(sf, sf / 30.0),
                'end_t': f2t.get(ef, ef / 30.0),
                'frames': run,
            })
        return out

    def all_action_ranges(self, pid):
        """Every action's range(s) across the whole clip, for the S2-style
        timeline. Returns [{action, label, start_t, end_t}] time-ordered.
        Label = action code, with an ordinal only when that action has >1 range
        (e.g. 'BS 1', 'BS 2')."""
        out = []
        for action in self.actions_for_patient(pid):
            ranges = self.get_ranges(pid, action)
            multi = len(ranges) > 1
            for r in ranges:
                out.append({
                    'action': action,
                    'label': f"{action} {r['idx']}" if multi else action,
                    'start_t': r['start_t'], 'end_t': r['end_t'],
                    'frames': list(r['frames']),   # ordered, for the defrag strip
                })
        out.sort(key=lambda x: x['start_t'])
        return out

    def actions_time_ordered(self, pid):
        """Actions in the order they first appear in the clip (timeline order),
        so the action map and natural traversal match the on-screen timeline."""
        seen = []
        for r in self.all_action_ranges(pid):
            if r['action'] not in seen:
                seen.append(r['action'])
        return seen

    def clip_span(self, pid, action):
        """(min_t, max_t) across ALL of the patient's frames — the full clip,
        for the overview timeline. Falls back to the action's span."""
        df = self.get_frame_df(pid)
        if df is None or 'timestamp' not in df.columns:
            frames = self.frames_for_action(pid, action)
            return (0.0, (max(frames) / 30.0) if frames else 1.0)
        t = df['timestamp'].astype(float)
        return (float(t.min()), float(t.max()))

    def au_values_at(self, pid, action, frame, aus):
        """{au: value} for the requested AUs at one frame (left side)."""
        df = self.get_frame_df(pid)
        if df is None:
            return {}
        sub = df[df['frame'] == frame]
        if len(sub) == 0:
            return {}
        sub = sub.iloc[0]
        return {au: float(sub[f'{au}_r']) for au in aus if f'{au}_r' in df.columns}

    def frame_time(self, pid, frame):
        df = self.get_frame_df(pid)
        if df is None or 'timestamp' not in df.columns:
            return frame / 30.0
        sub = df[df['frame'] == frame]
        if len(sub) == 0:
            return frame / 30.0
        return float(sub.iloc[0]['timestamp'])

    def inset_sum(self, pid, action, frame):
        """Sum of in-set AU intensities at one frame — the model's 'how strongly
        is this action being performed' signal (drives the confidence proxy)."""
        aus = self.inset.get(action, [])
        if not aus:
            return 0.0
        vals = self.au_values_at(pid, action, frame, aus)
        return float(sum(vals.values()))

    def inset_sums(self, pid, action):
        """{frame: in-set AU sum} for every frame of the action, computed in one
        pass (fast). Returns {} if the action has no in-set AUs."""
        aus = self.inset.get(action, [])
        df = self.get_frame_df(pid)
        if df is None or not aus:
            return {}
        cols = [f'{au}_r' for au in aus if f'{au}_r' in df.columns]
        sub = df[df['action'] == action]
        if len(sub) == 0 or not cols:
            return {}
        sums = sub[cols].sum(axis=1)
        return {int(f): float(s) for f, s in zip(sub['frame'], sums)}

    # ---------- auto-curator (validated per-action rules) ----------

    def _load_auto_params(self):
        """Load the leave-one-patient-out-CV-validated per-action rule params
        produced by s25_auto_curator.py. Missing file -> {} (legacy fallback)."""
        try:
            if config.AUTO_PARAMS_JSON.exists():
                return json.loads(config.AUTO_PARAMS_JSON.read_text())
        except Exception:
            pass
        return {}

    def _action_sub(self, pid, action):
        """Per-frame DataFrame for (pid, action), reading the STRONGER hemiface
        (higher key-AU peak) for the actions where CV showed it helps; otherwise
        the default side from get_frame_df. Returns a frame-sorted copy or None."""
        task_aus = config.FACS_TASK_AUS.get(action, [])
        if action in config.STRONGER_SIDE_ACTIONS and task_aus:
            best, best_pk = None, -1.0
            for side in ('left', 'right'):
                p = config.PER_FRAME_DIR / f'{pid}_{side}_mirrored_coded.csv'
                if not p.exists():
                    continue
                df = pd.read_csv(p)
                df['action'] = df['action'].astype(str).str.strip()
                s = df[df['action'] == action].sort_values('frame')
                cols = [f'{a}_r' for a in task_aus if f'{a}_r' in s.columns]
                pk = s[cols].sum(axis=1).max() if (len(s) and cols) else 0.0
                if pk > best_pk:
                    best_pk, best = pk, s.copy()
            if best is not None:
                return best
        df = self.get_frame_df(pid)
        if df is None:
            return None
        return df[df['action'] == action].sort_values('frame').copy()

    @staticmethod
    def _smooth(x, w):
        x = np.asarray(x, float)
        if w <= 1 or len(x) < w:
            return x
        return np.convolve(x, np.ones(int(w)) / int(w), mode='same')

    @staticmethod
    def _longest_run(mask):
        best, i = None, 0
        while i < len(mask):
            if mask[i]:
                j = i
                while j < len(mask) and mask[j]:
                    j += 1
                if best is None or (j - i) > (best[1] - best[0]):
                    best = (i, j)
                i = j
            else:
                i += 1
        return best

    def task_signal(self, pid, action, sub, signal_aus, closure=None):
        """Per-frame task signal aligned to `sub` (the reading side). Default = sum of
        signal_aus on `sub`. closure='min_both' = element-wise MIN of the per-side
        signal sums, so a frame scores high only when BOTH hemifaces show it — used
        for eye-closure tasks (ES/ET) so a frame with one eye still open scores low.
        The SAME method is called by the fit harness and by auto_keep_frames, so the
        fitted and deployed signals stay byte-identical."""
        def side_sum(df):
            c = [f'{a}_r' for a in signal_aus if f'{a}_r' in df.columns]
            return df[c].sum(axis=1).values if c else np.zeros(len(df))
        if closure == 'min_both':
            frames = list(sub['frame'].astype(int).values)
            per_side = []
            for side in ('left', 'right'):
                p = config.PER_FRAME_DIR / f'{pid}_{side}_mirrored_coded.csv'
                if not p.exists():
                    continue
                d = pd.read_csv(p)
                d['action'] = d['action'].astype(str).str.strip()
                s = d[d['action'] == action].sort_values('frame')
                if list(s['frame'].astype(int).values) != frames:
                    s = s.set_index('frame').reindex(frames).reset_index().fillna(0.0)
                per_side.append(side_sum(s))
            if len(per_side) == 2:
                return np.minimum(per_side[0], per_side[1])
            return per_side[0] if per_side else np.zeros(len(sub))
        return side_sum(sub)

    def auto_keep_frames(self, pid, action):
        """Validated auto-curator: apply the per-action rule fit by
        s25_auto_curator.py (leave-one-patient-out CV). Encodes the clinician's
        principles — keep near-maximal task performance, skip delayed onset, stop
        at early relaxation, reject blinks (except eye-closure tasks). Falls back
        to the legacy plateau if no params are loaded for the action."""
        params = (self.auto_params or {}).get(action)
        if params is None:
            return self._legacy_auto_keep(pid, action)

        sub = self._action_sub(pid, action)
        if sub is None or sub.empty:
            return self.frames_for_action(pid, action)
        frames = sub['frame'].astype(int).tolist()
        farr = np.array(frames)

        # Task signal: a validated focused AU subset (signal_aus) overrides the
        # full FACS in-set where a CV-confirmed dyad/triad tracks the expression
        # more cleanly (e.g. BS->AU10+AU25, SS->AU12+AU23). Side selection above
        # still uses the full set; only the keep-signal is focused.
        task_aus = params.get('signal_aus') or config.FACS_TASK_AUS.get(action, [])
        task = self.task_signal(pid, action, sub, task_aus, params.get('closure'))
        au45 = sub['AU45_r'].values if 'AU45_r' in sub.columns else np.zeros(len(sub))
        n = len(sub)
        pos = np.linspace(0, 1, n) if n > 1 else np.array([1.0])
        allcols = [f'{au}_r' for au in config.AU_ORDER if f'{au}_r' in sub.columns]
        total = sub[allcols].sum(axis=1).values
        eye = action in config.EYE_TASKS

        if action == 'BL':                       # rest: eyes-open, no blink, NON-SMILING
            keep = (au45 <= params['blink']) & (total <= params['rest_move'])
            scols = [f'{a}_r' for a in config.BL_SMILE_AUS if f'{a}_r' in sub.columns]
            smile = sub[scols].sum(axis=1).values if scols else np.zeros(n)
            neutral = smile < config.BL_SMILE_GATE
            if neutral.any():
                keep &= neutral                  # drop voluntary-smile frames
            elif smile.size:
                # smiles throughout -> keep the LEAST-smiling frames (caveated BL)
                keep &= (smile <= smile.min() + config.BL_SMILE_TOL)
        elif action in config.NO_PANEL_ACTIONS:  # position proxy
            keep = pos >= params['pos']
            if not eye:
                keep &= (au45 <= params['blink'])
        elif params.get('mode') == 'plateau':    # longest sustained run
            ss = self._smooth(task, params.get('smooth', 5))
            peak = ss.max() if ss.size else 0.0
            if peak <= 0:
                return frames
            rel = params['rel']
            if peak < params.get('abs_floor', 0.0):
                rel = max(rel, params.get('rel_strict', 0.75))
            run = self._longest_run(ss >= rel * peak)
            keep = np.zeros(n, dtype=bool)
            if run is not None:
                keep[run[0]:run[1]] = True
            if not eye:
                keep &= (au45 < params['blink'])
        else:                                    # frac-of-peak + position
            peak = task.max() if task.size else 0.0
            if peak <= 0:
                return frames
            keep = task >= params['frac'] * peak
            if params.get('pos', 0.0) > 0.0:
                keep &= (pos >= params['pos'])
            if not eye:
                keep &= (au45 <= params['blink'])

        return sorted(int(f) for f in farr[keep])

    def _legacy_auto_keep(self, pid, action):
        """Original plateau selector (fallback when no validated params exist)."""
        if action == 'BL' or not self.inset.get(action):
            return self.frames_for_action(pid, action)
        sums = self.inset_sums(pid, action)
        if not sums:
            return self.frames_for_action(pid, action)
        peak = max(sums.values())
        if peak <= 0:
            return self.frames_for_action(pid, action)
        thr = max(config.AUTO_KEEP_FRAC * peak, config.AUTO_KEEP_FLOOR)
        return sorted(f for f, s in sums.items() if s >= thr)

    def low_confidence_frames(self, pid, action, band=0.15):
        """Frames whose in-set activation sits near the plateau keep/reject
        threshold — the model is least certain about these, so they get the
        review flag. Threshold = AUTO_KEEP_FRAC * peak. Low-confidence if within
        +/- band*peak of it."""
        sums = self.inset_sums(pid, action)
        if not sums:
            return set()
        peak = max(sums.values())
        if peak <= 0:
            return set()
        thr = max(config.AUTO_KEEP_FRAC * peak, config.AUTO_KEEP_FLOOR)
        lo = set()
        for f, s in sums.items():
            if abs(s - thr) <= band * peak:
                lo.add(f)
        return lo

    def label_aus_for_action(self, action):
        """In-set AUs + always-show (AU45) for label display."""
        aus = list(self.inset.get(action, []))
        for au in config.ALWAYS_SHOW_AU:
            if au not in aus:
                aus.append(au)
        return aus

    # ---------- face crop ----------

    def _ensure_cascade(self):
        if self._cascade is None:
            self._cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return self._cascade

    def get_crop_box(self, pid):
        """(x, y, w, h) face-region crop box in source-video pixels. Detected
        once per patient (head ~stationary across tasks); cached."""
        with self._cache_lock:
            if pid in self._crop_boxes:
                return self._crop_boxes[pid]
        box = self._detect_face_box(pid)   # slow detection outside the lock
        with self._cache_lock:
            self._crop_boxes[pid] = box
        return box

    def _detect_face_box(self, pid):
        video = config.SOURCE_VIDEO_DIR / f'{pid}_source_coded.mp4'
        if not video.exists():
            return None
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cascade = self._ensure_cascade()
        boxes = []
        for k in range(config.FACE_DETECT_SAMPLES):
            frac = (k + 1) / (config.FACE_DETECT_SAMPLES + 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac))
            ret, fr = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(120, 120))
            if len(faces):
                boxes.append(max(faces, key=lambda b: b[2] * b[3]))
        cap.release()
        if not boxes:
            # fallback: face-centered box at thumbnail aspect in upper-middle
            fx, fy, fw, fh = int(W * 0.18), int(H * 0.12), int(W * 0.64), int(W * 0.64)
        else:
            arr = np.array(boxes)
            fx, fy, fw, fh = np.median(arr, axis=0).astype(int)
        # margins around the Haar face box (keep forehead/brows + chin)
        x0 = fx - fw * config.FACE_EXPAND_SIDE
        x1 = fx + fw + fw * config.FACE_EXPAND_SIDE
        y0 = fy - fh * config.FACE_EXPAND_TOP
        y1 = fy + fh + fh * config.FACE_EXPAND_BOTTOM
        w0, h0 = x1 - x0, y1 - y0
        # Adjust to the thumbnail aspect ratio so the crop fills the cell with
        # no letterbox (face stays centered).
        target = config.THUMB_W / config.THUMB_H
        if w0 / h0 < target:           # too tall -> widen
            new_w = h0 * target
            x0 -= (new_w - w0) / 2
            w0 = new_w
        else:                          # too wide -> heighten
            new_h = w0 / target
            y0 -= (new_h - h0) / 2
            h0 = new_h
        x = max(0, int(round(x0)))
        y = max(0, int(round(y0)))
        w = min(W - x, int(round(w0)))
        h = min(H - y, int(round(h0)))
        return (x, y, w, h)

    def _crop_face(self, img, pid):
        box = self.get_crop_box(pid)
        if box is None:
            return img
        x, y, w, h = box
        h_img, w_img = img.shape[:2]
        x2, y2 = min(w_img, x + w), min(h_img, y + h)
        if x2 <= x or y2 <= y:
            return img
        return img[y:y2, x:x2]

    # ---------- thumbnails ----------

    def get_thumbnail(self, pid, frame):
        """Return a face-cropped RGB ndarray (resized) for the source frame."""
        key = (pid, frame)
        with self._cache_lock:
            if key in self._thumb_cache:
                self._thumb_cache.move_to_end(key)
                return self._thumb_cache[key]
        video = config.SOURCE_VIDEO_DIR / f'{pid}_source_coded.mp4'
        img = self._read_frame(video, frame)
        if img is not None:
            img = self._crop_face(img, pid)
            img = self._resize_fill(img, config.THUMB_W, config.THUMB_H)
        with self._cache_lock:
            self._thumb_cache[key] = img
            while len(self._thumb_cache) > self._thumb_cache_cap:
                self._thumb_cache.popitem(last=False)
        return img

    def preload_action_thumbnails(self, pid, action, progress_cb=None):
        """Sequentially read the action's frame range once (fast vs per-frame seek)."""
        frames = self.frames_for_action(pid, action)
        if not frames:
            return
        with self._cache_lock:
            need = [f for f in frames if (pid, f) not in self._thumb_cache]
        if not need:
            return
        video = config.SOURCE_VIDEO_DIR / f'{pid}_source_coded.mp4'
        if not video.exists():
            return
        lo, hi = min(need), max(need)
        need_set = set(need)
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        fidx = lo
        done = 0
        while fidx <= hi:
            ret, fr = cap.read()
            if not ret:
                break
            if fidx in need_set:
                rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                rgb = self._crop_face(rgb, pid)
                rgb = self._resize_fill(rgb, config.THUMB_W, config.THUMB_H)
                with self._cache_lock:
                    self._thumb_cache[(pid, fidx)] = rgb
                    self._thumb_cache.move_to_end((pid, fidx))
                done += 1
                if progress_cb:
                    progress_cb(done, len(need))
            fidx += 1
        cap.release()
        with self._cache_lock:
            while len(self._thumb_cache) > self._thumb_cache_cap:
                self._thumb_cache.popitem(last=False)

    @staticmethod
    def _read_frame(video, frame_idx):
        if not video.exists():
            return None
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
        ret, fr = cap.read()
        cap.release()
        if not ret or fr is None:
            return None
        return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _resize_keep(img, w, h):
        ih, iw = img.shape[:2]
        scale = min(w / iw, h / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _resize_fill(img, w, h):
        """Cover the w x h cell (scale to fill) then center-crop the overflow.
        Guarantees the thumbnail fills the cell with no letterbox."""
        ih, iw = img.shape[:2]
        if ih == 0 or iw == 0:
            return img
        scale = max(w / iw, h / ih)
        nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
        r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        x0 = max(0, (nw - w) // 2)
        y0 = max(0, (nh - h) // 2)
        return r[y0:y0 + h, x0:x0 + w]

    # ---------- curation state ----------

    def _load_curation(self):
        if config.CURATION_JSON.exists():
            try:
                with open(config.CURATION_JSON) as f:
                    return json.load(f)
            except Exception:
                pass
        return {'version': 1, 'patients': {}}

    def save_curation(self):
        """Atomic write: serialize to a temp file, fsync, then os.replace so a
        crash mid-write can never truncate/corrupt the real curation file."""
        config.CURATION_JSON.parent.mkdir(parents=True, exist_ok=True)
        tmp = config.CURATION_JSON.with_name(config.CURATION_JSON.name + '.tmp')
        try:
            with open(tmp, 'w') as f:
                json.dump(self.curation, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, config.CURATION_JSON)
        except Exception:
            # serialization/write failed: leave the real file untouched and
            # don't leave a half-written temp behind.
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    def _snapshot_curation(self, tag):
        """Copy the last-saved curation file to a timestamped backup before a
        destructive op (e.g. a re-score reconcile reset), so the prior curation is
        always recoverable. Keeps the newest 30 snapshots. Best-effort (never raises
        into the caller)."""
        try:
            if not config.CURATION_JSON.exists():
                return
            bdir = config.CURATION_JSON.parent / "curation_backups"
            bdir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe = ''.join(c if (c.isalnum() or c in '._-') else '_' for c in str(tag))
            dst = bdir / f"s25_curation.{safe}.{ts}.json"
            if not dst.exists():
                dst.write_text(config.CURATION_JSON.read_text())
            snaps = sorted(bdir.glob("s25_curation.*.json"))
            for old in snaps[:-30]:
                try:
                    old.unlink()
                except OSError:
                    pass
            print(f"[snapshot] curation backed up -> {dst.name}")
        except Exception as e:
            print(f"[snapshot] failed ({tag}): {e}")

    # ---- baseline (BL) recovery ----
    # Control resting-BL gate: a window whose mean total-AU is at/below the
    # control cohort's BL p95 is a genuine neutral rest; above it, the window is
    # the patient's lowest-tone available but carries resting tone ('elevated').
    BL_NEUTRAL_GATE = 5.5     # control BL total-AU p95 (recompute as controls grow)
    BL_MIN_LEN = 6            # shortest acceptable BL window (frames)
    BL_AU45_OPEN = 0.7        # a BL frame must be EYES-OPEN (AU45 below this)
    BL_HEAD_SEARCH = 45       # how far into the clip the 'start' tier searches
    BL_TONE_TOL = 1.5         # clip-start tier prefers the EARLIEST eyes-open window
                              # within this much tone of the quietest (so BL sits at
                              # the true start unless a later window is clearly calmer)

    def _control_bl_gate(self):
        return getattr(self, '_bl_gate_cache', self.BL_NEUTRAL_GATE)

    def _aus_sum(self, df, aus):
        cols = [f'{a}_r' for a in aus if f'{a}_r' in df.columns]
        return df[cols].sum(axis=1).values if cols else np.zeros(len(df))

    def choose_baseline(self, pid):
        """Pick the resting-baseline window (the permanent baseline selector; tuned
        by config BL_* constants). The baseline is the patient at REST:

          1. OPENING (default): the quietest eyes-open window in the opening rest —
             frames before the FIRST task's real onset. Onset = where that task's
             defining AU signal actually ramps (BL_ONSET_FRAC of its peak), so
             leading frames the coder mis-labelled as the first task (patient still
             at rest) are reclaimed into the baseline.
          2. LATER (exception): if the opening is contaminated — smiling
             (>= BL_CONTAM_SMILE) or active (tone >= BL_CONTAM_TONE) — AND a
             materially quieter later window exists (tone lower by >= BL_SWITCH_MARGIN
             and not more smiling), use that instead (the heavy-smiler-at-the-start
             case). Later candidates are eyes-open, brow-quiet, uncoded-or-BL.

        Windows are SCORED on their quietest BL_SEED_WIN-frame run, then widened only
        across frames within BL_EXTEND_TONE of that seed (so a wider coded window can
        never inflate the score). 'tone' excludes AU45 so eye-closure can't look
        quiet. Returns dict(frames, start, end, tone_excl_blink, quality, decision,
        reclaim_from) or None. quality: 'smiling' (window smiles >= gate) else
        'neutral'/'elevated' by the control rest gate."""
        df = self.get_frame_df(pid)
        if df is None or df.empty:
            return None
        df = df.sort_values('frame').reset_index(drop=True)
        df['action'] = df['action'].astype(str).str.strip()
        aucols = [f'{a}_r' for a in config.AU_ORDER if f'{a}_r' in df.columns]
        fr = df['frame'].astype(int).values
        au45 = df['AU45_r'].values if 'AU45_r' in df.columns else np.zeros(len(df))
        tone = df[aucols].sum(axis=1).values - au45     # exclude blink from 'activity'
        smile = self._aus_sum(df, config.BL_SMILE_AUS)
        brow = self._aus_sum(df, config.BL_BROW_AUS)
        oral = self._aus_sum(df, config.BL_ORAL_AUS)
        acts = df['action'].values
        eo = au45 < config.BL_OPEN_EYE

        def best_run(mask, wlen=config.BL_SEED_WIN):
            """Lowest-tone eyes-open run in `mask`. Score = the quietest `wlen`-window
            (seed); widen only across frames within BL_EXTEND_TONE of it so a wider
            coded window never inflates the score. Returns (lo, hi, seed, smile)."""
            best = None
            for s in range(0, len(tone) - wlen + 1):
                if mask[s:s + wlen].all():
                    m = float(tone[s:s + wlen].mean())
                    if best is None or m < best[0]:
                        best = (m, s)
            if best is None:
                return None
            seed, s = best
            lo, hi = s, s + wlen
            while lo > 0 and mask[lo - 1] and tone[lo - 1] <= seed + config.BL_EXTEND_TONE \
                    and (hi - lo) < config.BL_EXTEND_CAP:
                lo -= 1
            while hi < len(tone) and mask[hi] and tone[hi] <= seed + config.BL_EXTEND_TONE \
                    and (hi - lo) < config.BL_EXTEND_CAP:
                hi += 1
            return lo, hi, seed, float(smile[lo:hi].mean())

        # ---- first task's REAL onset = the opening-rest boundary (reclaim before it)
        first_idx = {}
        for i, a in enumerate(acts):
            if a not in ('BL', 'nan', ''):
                first_idx.setdefault(a, i)
        onset = len(fr)
        if first_idx:
            ft = min(first_idx, key=lambda a: first_idx[a])
            fs = first_idx[ft]
            fe = max(i for i, a in enumerate(acts) if a == ft)
            sa = [f'{x}_r' for x in config.FACS_TASK_AUS.get(ft, []) if f'{x}_r' in df.columns]
            sig = df[sa].sum(axis=1).values if sa else tone
            hi = min(fe, fs + config.BL_ONSET_SEARCH)
            peak = sig[fs:hi + 1].max() if hi >= fs else 0.0
            onset = fs
            if peak > 0.5:
                thr = config.BL_ONSET_FRAC * peak
                for i in range(fs, hi + 1):
                    if sig[i] >= thr and (i + 1 > hi or sig[i + 1] >= thr):
                        onset = i
                        break

        # ---- opening candidate (before onset). Prefer a full seed; accept a SHORTER
        # window when the pre-task rest is brief (a clip that opens straight into the
        # first task, e.g. RE coded from frame 0); else fall back to the current BL.
        om = eo.copy()
        om[onset:] = False
        ob = best_run(om) or best_run(om, config.BL_MIN_LEN) \
            or best_run(eo & (acts == 'BL'))
        # ---- later candidate (eyes-open, brow-quiet, ORAL-quiet, uncoded-or-BL,
        # at/after onset). The oral gate rejects off-panel oral actions (tongue/lip-
        # press/pucker) that tone can't see but that shadow in AU17+AU23.
        lm = eo & (brow < config.BL_BROW_MAX) & (oral < config.BL_ORAL_MAX) \
            & np.array([a in ('nan', '', 'BL') for a in acts])
        lm[:onset] = False
        # Off-panel guard: residual cheek-puff/pucker/platysma/lip activity after an
        # off-panel-target task is invisible to tone, so a window there can look quiet
        # while the patient isn't at rest. Drop every frame from an off-panel task
        # onward until the next ON-panel task re-establishes a tone-visible reference.
        last = None
        for i, a in enumerate(acts):
            if a not in ('BL', 'nan', ''):
                last = a
            if last in config.BL_OFFPANEL_ACTIONS:
                lm[i] = False
        lb = best_run(lm)

        if ob is None:
            # No usable opening rest (clip opens straight into the first task). Fall
            # back to a later quiet window so every patient still gets a baseline; if
            # there is no eyes-open window anywhere, none can be synthesized.
            if lb is None:
                return None
            win, decision = lb, 'later'
        else:
            ot, osm = ob[2], ob[3]
            # The opening is the default. Switch to a later window only if the opening
            # is contaminated (smiling or active) AND a later window is materially
            # quieter AND that later window is CLEAN — either a neutral rest (smile <
            # gate) or much less smiling than an egregiously-smiling opening. The clean
            # test stops a switch to a still-smiling later (8270 -> 906-916, smile 2.7)
            # while allowing 4036 (4.8 -> 2.2) and neutral laters (20250225 -> 80-87).
            contaminated = (osm >= config.BL_CONTAM_SMILE) or (ot >= config.BL_CONTAM_TONE)
            much_quieter = lb is not None and lb[2] <= ot - config.BL_SWITCH_MARGIN \
                and lb[3] <= osm + 0.3
            later_clean = lb is not None and (
                lb[3] < config.BL_SMILE_GATE
                or (osm >= config.BL_CONTAM_SMILE
                    and lb[3] <= osm - config.BL_SWITCH_SMILE_DROP))
            win, decision = (lb, 'later') if (contaminated and much_quieter and later_clean) \
                else (ob, 'opening')
        lo, hi, seed_tone, win_smile = win
        bl_frames = [int(f) for f in fr[lo:hi]]
        reclaim = sorted(set(acts[lo:hi]) - {'BL', 'nan', ''})
        if win_smile >= config.BL_SMILE_GATE:
            quality = 'smiling'
        else:
            quality = 'neutral' if seed_tone <= self._control_bl_gate() else 'elevated'
        return {'frames': bl_frames, 'start': bl_frames[0], 'end': bl_frames[-1],
                'tone_excl_blink': round(float(seed_tone), 2), 'quality': quality,
                'decision': decision, 'reclaim_from': reclaim}

    def recover_baseline(self, pid):
        """Reconcile entry point: synthesize a baseline ONLY when none is coded yet
        (the every-patient-must-have-a-BL rule). Delegates to choose_baseline."""
        if 'BL' in self.actions_for_patient(pid):
            return None
        return self.choose_baseline(pid)

    def apply_baseline(self, pid, choice):
        """Persist a chosen baseline into BOTH hemiface CSVs: un-code the current BL,
        code the chosen window (reclaiming ONLY the mis-coded first-task frames named
        in choice['reclaim_from'] — never steals any other action), then reset the BL
        node and each reclaimed-task node for re-curation. AU columns are hash-verified
        unchanged; a one-time .prebaseline backup is taken. Returns True on success."""
        if not choice:
            return False
        frames = set(choice['frames'])
        reclaim = {str(a).strip() for a in (choice.get('reclaim_from') or ())}
        allowed = {'nan'} | reclaim
        side_bl = {}
        wrote = False
        for side in ('left', 'right'):
            csv = config.PER_FRAME_DIR / f'{pid}_{side}_mirrored_coded.csv'
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            if 'action' not in df.columns:
                continue
            aucols = [c for c in df.columns
                      if c.endswith('_r') or c.endswith('_r_static')]
            pre = hashlib.md5(pd.util.hash_pandas_object(
                df[aucols], index=True).values.tobytes()).hexdigest()
            bk = config.PER_FRAME_DIR / f'{pid}_{side}_mirrored_coded.csv.prebaseline'
            if not bk.exists():
                df.to_csv(bk, index=False)
            act = df['action'].astype(str).str.strip()
            df.loc[act == 'BL', 'action'] = np.nan                 # un-code old BL
            act2 = df['action'].astype(str).str.strip()
            mask = df['frame'].isin(frames) & act2.isin(allowed)   # code new (never steals)
            df.loc[mask, 'action'] = 'BL'
            post = hashlib.md5(pd.util.hash_pandas_object(
                df[aucols], index=True).values.tobytes()).hexdigest()
            assert pre == post, f"AU columns changed writing BL for {pid}/{side}!"
            df.to_csv(csv, index=False)
            side_bl[side] = set(int(f) for f in
                                df.loc[df['action'].astype(str).str.strip() == 'BL', 'frame'])
            wrote = True
        if not wrote:
            return False
        if len(side_bl) == 2:
            assert side_bl['left'] == side_bl['right'], \
                f"hemiface BL mismatch for {pid}: {side_bl}"
        with self._cache_lock:
            self._frame_cache.pop(pid, None)
        self.curation.setdefault('bl_quality', {})[pid] = choice['quality']
        self.reset_action_to_auto(pid, 'BL')
        for a in reclaim:                       # reclaimed task shrank -> re-curate
            if a in self.actions_for_patient(pid):
                self.reset_action_to_auto(pid, a)
        return True

    def write_baseline(self, pid, bl_frames, quality, reclaim_from=None):
        """Persist a synthesized BL into BOTH hemiface v1316 CSVs: set action='BL'
        on the given frames. AU columns are untouched (hash-verified). Records the
        BL quality ('neutral'/'elevated') in curation meta. Returns True on success.

        reclaim_from: optional set of action codes whose frames may be reclaimed for
        BL. Used ONLY for the clip-start tier, where S2 sometimes mis-codes the
        opening rest as the first task (e.g. 20250409: opening rest coded 'RE' with
        no real brow-raise). Because bl_frames is a single clip-start window, only
        those leading frames are reclaimed; any genuine later instance of the donor
        action is untouched. The donor action then auto-resets for re-curation via
        the frame-signature reconcile (its coded set shrank)."""
        reclaim = {str(a).strip() for a in (reclaim_from or ())}
        allowed = {'nan'} | reclaim
        wrote = False
        for side in ('left', 'right'):
            csv = config.PER_FRAME_DIR / f'{pid}_{side}_mirrored_coded.csv'
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            if 'action' not in df.columns:
                continue
            aucols = [c for c in df.columns
                      if c.endswith('_r') or c.endswith('_r_static')]
            pre = hashlib.md5(pd.util.hash_pandas_object(
                df[aucols], index=True).values.tobytes()).hexdigest()
            act = df['action'].astype(str).str.strip()
            blset = set(bl_frames)
            mask = df['frame'].isin(blset) & act.isin(allowed)
            if not mask.any():
                continue
            # backup once
            bk = config.PER_FRAME_DIR / f'{pid}_{side}_mirrored_coded.csv.blbak'
            if not bk.exists():
                df.to_csv(bk, index=False)
            df.loc[mask, 'action'] = 'BL'
            post = hashlib.md5(pd.util.hash_pandas_object(
                df[aucols], index=True).values.tobytes()).hexdigest()
            assert pre == post, f"AU columns changed writing BL for {pid}/{side}!"
            df.to_csv(csv, index=False)
            wrote = True
        if wrote:
            with self._cache_lock:
                self._frame_cache.pop(pid, None)   # force reload of corrected data
            meta = self.curation.setdefault('bl_quality', {})
            meta[pid] = quality
        return wrote

    def bl_quality(self, pid):
        return self.curation.get('bl_quality', {}).get(pid)

    def frame_signature(self, pid, action):
        """A stable signature of the action's CURRENT coded frame set. If a
        re-score (or any data regeneration) changes which frames belong to the
        action, this signature changes — which is how reconcile_patient() knows
        to reset the action for re-curation."""
        frames = self.frames_for_action(pid, action)
        h = hashlib.md5((','.join(str(f) for f in frames)).encode()).hexdigest()
        return f"{len(frames)}:{h[:16]}"

    # ---- "needs merge" flag (patient awaiting the manual v1316 AU-swap after an
    # S2 re-score). Stored at curation top-level so it persists across restarts.
    def _needs_merge_set(self):
        return set(self.curation.setdefault('needs_merge', []))

    def set_needs_merge(self, pid, on=True):
        s = self._needs_merge_set()
        s.add(pid) if on else s.discard(pid)
        self.curation['needs_merge'] = sorted(s)

    def needs_merge(self, pid):
        return pid in self._needs_merge_set()

    # S1 mirrored-CSV schema (what S2 consumes): a strict subset of v1316's columns.
    _S1_MIRRORED_COLS = ['frame', 'timestamp', 'success',
        'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r',
        'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r',
        'AU25_r', 'AU26_r', 'AU45_r']

    def ensure_mirrored_csvs(self, pid):
        """S2 needs {pid}_{side}_mirrored.csv (S1 AU format) in Combined Data to load
        a patient for re-scoring. If absent, DERIVE them on demand from the v1316
        per-frame data (the 20-col S1 schema is a strict subset of v1316), so any
        patient is re-scorable without pre-generating all of them. Never overwrites
        an existing file. The derived AUs are v1316 values — consistent, since the
        re-score merge keeps v1316 AUs. Returns True if the CSVs now exist, False if
        there's no v1316 data to derive from."""
        cd = config.S1_COMBINED_DIR
        if (cd / f"{pid}_left_mirrored.csv").exists() or \
           (cd / f"{pid}_right_mirrored.csv").exists():
            return True
        for side in ('left', 'right'):
            src = config.PER_FRAME_DIR / f'{pid}_{side}_mirrored_coded.csv'
            dst = cd / f'{pid}_{side}_mirrored.csv'
            if dst.exists() or not src.exists():
                continue
            try:
                df = pd.read_csv(src)
            except Exception:
                continue
            if any(c not in df.columns for c in self._S1_MIRRORED_COLS):
                continue
            df[self._S1_MIRRORED_COLS].to_csv(dst, index=False)
            print(f"[re-score] derived {dst.name} from v1316 (AUs = v1316 values)")
        return (cd / f"{pid}_left_mirrored.csv").exists() or \
               (cd / f"{pid}_right_mirrored.csv").exists()

    def import_rescored_actions(self, pid, s2_outputs=None):
        """PRODUCTION re-score merge. Copy the re-scored ACTION column from S2's
        per-frame output into BOTH hemiface v1316 CSVs, PRESERVING v1316's validated
        AU columns (pilot policy: keep v1316 AUs — S2's own output is dynamic-AU only
        and lacks the static cols). Frames must align 1:1 or the merge is refused
        (never corrupt). Each v1316 CSV is backed up once (.premerge) before its
        first merge. Returns (ok: bool, info: str). After this, the caller runs
        reconcile_patient() so every action whose frames changed re-curates."""
        src = {}
        for o in (s2_outputs or []):
            o = str(o)
            if o.endswith('_left_mirrored_coded.csv'):
                src['left'] = Path(o)
            elif o.endswith('_right_mirrored_coded.csv'):
                src['right'] = Path(o)
        s2o_dir = getattr(config, 'S2O_DIR', None) or getattr(config, 'SOURCE_VIDEO_DIR', None)
        for side in ('left', 'right'):
            if src.get(side) is None and s2o_dir is not None:
                src[side] = Path(s2o_dir) / f'{pid}_{side}_mirrored_coded.csv'
        # Action coding is per-VIDEO — identical across hemifaces by design — so use a
        # SINGLE canonical action column for BOTH sides, taken from the most-recently
        # written S2 output. S2 sometimes re-saves only the side that was on screen, so
        # merging each side from its own file can split-brain (one v1316 side keeps
        # stale coding; the curator reads the stale side). Picking the newest output
        # and applying it to both sides makes that impossible.
        avail = [p for p in src.values() if p is not None and Path(p).exists()]
        if not avail:
            return False, "S2 output missing for both sides"
        canon = max(avail, key=lambda p: Path(p).stat().st_mtime)
        dfc = pd.read_csv(canon)
        canon_frames = list(dfc['frame'].astype(int))
        canon_act = dfc['action'].values                  # raw (preserves NaN repr)
        canon_norm = dfc['action'].astype(str).str.strip().values
        changed_total, sides = 0, []
        for side in ('left', 'right'):
            v3 = config.PER_FRAME_DIR / f'{pid}_{side}_mirrored_coded.csv'
            if not v3.exists():
                return False, f"v1316 file missing for {side}"
            dfv = pd.read_csv(v3)
            if list(dfv['frame'].astype(int)) != canon_frames:
                return False, (f"frame mismatch on {side} "
                               f"(v1316 {len(dfv)} vs S2 {len(canon_frames)}) — not merged")
            aucols = [c for c in dfv.columns
                      if c.endswith('_r') or c.endswith('_r_static')]
            pre = hashlib.md5(pd.util.hash_pandas_object(
                dfv[aucols], index=True).values.tobytes()).hexdigest()
            old_act = dfv['action'].astype(str).str.strip().values
            changed_total += int((canon_norm != old_act).sum())
            bk = v3.with_suffix('.csv.premerge')
            if not bk.exists():
                dfv.to_csv(bk, index=False)
            dfv['action'] = canon_act                     # same coding to BOTH sides
            post = hashlib.md5(pd.util.hash_pandas_object(
                dfv[aucols], index=True).values.tobytes()).hexdigest()
            assert pre == post, f"AU columns changed merging actions for {pid}/{side}!"
            dfv.to_csv(v3, index=False)
            sides.append(side)
        with self._cache_lock:
            self._frame_cache.pop(pid, None)   # force reload of merged data
        return True, f"{changed_total} action cells -> both sides from {canon.name}"

    def reconcile_patient(self, pid):
        """PRODUCTION rule: any action whose coded frame set has changed since it
        was last curated is reset for the auto-curator + fresh human review.
        Returns the list of actions that were reset.

        Detection is signature-based (no diffing): each node stores the frame
        signature it was curated against. On load we compare to the current
        signature. Existing nodes without a signature are BACKFILLED with the
        current one (not reset) so prior curation never false-resets.

        Also enforces the 'every patient must have a BL' rule: if no BL is coded,
        synthesize one from the lowest-tone window (control-gated neutral/elevated)
        and write it into the per-frame data before reconciling."""
        reset = []
        # BL recovery FIRST (may add a BL action the reconcile then picks up)
        if 'BL' not in self.actions_for_patient(pid):
            prop = self.recover_baseline(pid)
            if prop and self.write_baseline(pid, prop['frames'], prop['quality'],
                                            reclaim_from=prop.get('reclaim_from')):
                reset.append('BL')   # report the synthesized BL as new work
        pnode = self.curation['patients'].get(pid)
        if not isinstance(pnode, dict):
            return reset
        snapped = False
        for action in self.actions_for_patient(pid):
            node = pnode.get(action)
            if not isinstance(node, dict) or 'kept' not in node:
                continue
            cur_sig = self.frame_signature(pid, action)
            stored = node.get('frame_sig')
            if stored is None:
                node['frame_sig'] = cur_sig          # migrate: trust as-is
                continue
            if stored == cur_sig:
                continue
            # about to reset curation -> snapshot the whole curation file ONCE
            # (the last-saved, pre-reset state) so any reset is recoverable.
            if not snapped:
                self._snapshot_curation(f"pre_reset_{pid}")
                snapped = True
            # frames changed -> reset selection + status, PRESERVE flags/note
            keep_flags = list(node.get('flags', []))
            keep_note = node.get('note', '')
            auto = self.auto_keep_frames(pid, action)
            node['kept'] = sorted(auto)
            node['auto_kept'] = list(auto)
            node['confirmed'] = []
            node['status'] = 'todo'
            node['flags'] = keep_flags
            node['note'] = keep_note
            node['frame_sig'] = cur_sig
            node.pop('curated_at', None)
            reset.append(action)
        # If a reset fired, the merge that changed the frames has landed -> the
        # patient no longer "needs merge".
        if reset and self.needs_merge(pid):
            self.set_needs_merge(pid, False)
        return reset

    def get_action_state(self, pid, action):
        """Single-axis curation state for (pid, action). Every frame starts on
        the model's prediction (auto-window = kept). Frames are either KEPT
        (representative) or rejected (not characteristic — derived as the
        complement). 'confirmed' tracks frames the human has reviewed."""
        pnode = self.curation['patients'].setdefault(pid, {})
        node = pnode.get(action)
        # Initialize, or migrate an old-schema node (included/excluded -> kept).
        if node is None or 'kept' not in node:
            auto = self.auto_keep_frames(pid, action)   # plateau selection
            kept = node['included'] if (node and 'included' in node) else list(auto)
            pnode[action] = {
                'status': (node.get('status', 'todo') if node else 'todo'),
                'kept': sorted(kept),
                'auto_kept': list(auto),
                'confirmed': [],
                'flags': [],
                'note': '',
                'frame_sig': self.frame_signature(pid, action),
            }
        node = pnode[action]
        node.setdefault('flags', [])   # migrate older nodes
        node.setdefault('note', '')
        node.setdefault('frame_sig', self.frame_signature(pid, action))
        return node

    def set_kept(self, pid, action, frame, kept):
        st = self.get_action_state(pid, action)
        keptset = set(st['kept'])
        conf = set(st['confirmed'])
        if kept:
            keptset.add(frame)
        else:
            keptset.discard(frame)
        conf.add(frame)   # touching a frame confirms it
        st['kept'] = sorted(keptset)
        st['confirmed'] = sorted(conf)

    def set_kept_bulk(self, pid, action, frames, kept):
        st = self.get_action_state(pid, action)
        keptset = set(st['kept'])
        conf = set(st['confirmed'])
        for f in frames:
            if kept:
                keptset.add(f)
            else:
                keptset.discard(f)
            conf.add(f)
        st['kept'] = sorted(keptset)
        st['confirmed'] = sorted(conf)

    def accept_all_auto(self, pid, action):
        """Confirm the model's prediction as-is for the whole action."""
        st = self.get_action_state(pid, action)
        st['kept'] = list(st['auto_kept'])
        st['confirmed'] = sorted(self.frames_for_action(pid, action))

    def clear_action(self, pid, action):
        st = self.get_action_state(pid, action)
        st['kept'] = []
        st['confirmed'] = sorted(self.frames_for_action(pid, action))

    def counts(self, pid, action):
        """(kept, rejected, to_review) for the count cards."""
        st = self.get_action_state(pid, action)
        all_frames = self.frames_for_action(pid, action)
        kept = len(st['kept'])
        rejected = len(all_frames) - kept
        lowconf = self.low_confidence_frames(pid, action)
        confirmed = set(st['confirmed'])
        to_review = len(lowconf - confirmed)
        return kept, rejected, to_review

    def peak_inset(self, pid, action):
        sums = self.inset_sums(pid, action)
        return max(sums.values()) if sums else 0.0

    def snapshot(self, pid, action):
        """Deep-ish copy of the action state for undo."""
        st = self.get_action_state(pid, action)
        return {'kept': list(st['kept']),
                'confirmed': list(st['confirmed']),
                'status': st.get('status', 'todo'),
                'flags': list(st.get('flags', [])),
                'note': st.get('note', '')}

    def restore(self, pid, action, snap):
        st = self.get_action_state(pid, action)
        st['kept'] = list(snap['kept'])
        st['confirmed'] = list(snap['confirmed'])
        st['status'] = snap.get('status', 'todo')
        st['flags'] = list(snap.get('flags', []))
        st['note'] = snap.get('note', '')

    def reset_action_to_auto(self, pid, action):
        """Reset ONE action's curation back to the auto-curator output: kept=auto,
        status=todo, confirmed cleared, curated_at dropped. Flags + note are
        PRESERVED (clinical observations). Returns the auto keep set. Used by the
        'reset curation' button (caller snapshots for undo first)."""
        st = self.get_action_state(pid, action)
        auto = self.auto_keep_frames(pid, action)
        st['kept'] = sorted(auto)
        st['auto_kept'] = list(auto)
        st['confirmed'] = []
        st['status'] = 'todo'
        st['frame_sig'] = self.frame_signature(pid, action)
        st.pop('curated_at', None)
        return auto

    def mark_status(self, pid, action, status):
        st = self.get_action_state(pid, action)
        st['status'] = status
        st['curated_at'] = datetime.now(timezone.utc).isoformat()

    # Task-performance flags (per patient+action): the patient either did not
    # perform the task as instructed, or performed it in an abnormal manner.
    FLAG_NOT_PERFORMED = 'not_performed'
    FLAG_ABNORMAL = 'abnormal'

    def toggle_flag(self, pid, action, flag):
        st = self.get_action_state(pid, action)
        flags = set(st.get('flags', []))
        if flag in flags:
            flags.discard(flag)
        else:
            flags.add(flag)
        st['flags'] = sorted(flags)
        return flag in flags

    def get_flags(self, pid, action):
        return list(self.get_action_state(pid, action).get('flags', []))

    def set_flag(self, pid, action, flag, on=True):
        """Idempotently set/clear a flag (vs toggle_flag which flips)."""
        st = self.get_action_state(pid, action)
        fl = set(st.get('flags', []))
        fl.add(flag) if on else fl.discard(flag)
        st['flags'] = sorted(fl)

    def set_note(self, pid, action, text):
        self.get_action_state(pid, action)['note'] = (text or '').strip()

    def get_note(self, pid, action):
        return self.get_action_state(pid, action).get('note', '')

    def action_status(self, pid, action):
        pnode = self.curation['patients'].get(pid, {})
        if action in pnode:
            return pnode[action].get('status', 'todo')
        return 'todo'

    def patient_progress(self, pid):
        """(n_done, n_total) over ALL present actions — matches the save & next
        traversal, so a patient reads 'done' exactly when every window has been
        reviewed (not just the analyzed subset)."""
        actions = self.actions_for_patient(pid)
        if not actions:
            return (0, 0)
        done = sum(1 for a in actions if self.action_status(pid, a) == 'done')
        return (done, len(actions))

    def patient_status(self, pid):
        done, total = self.patient_progress(pid)
        if total == 0:
            return 'todo'
        if done == 0:
            return 'todo'
        if done == total:
            return 'done'
        return 'partial'

    def overall_progress(self):
        """(n_patients_done, n_patients_total)."""
        done = sum(1 for p in self.patients if self.patient_status(p) == 'done')
        return (done, len(self.patients))

    # ---------- export ----------

    def export_csv(self):
        rows = []
        for pid, pnode in self.curation['patients'].items():
            for action, st in pnode.items():
                kept = st.get('kept', [])
                auto = st.get('auto_kept', [])
                rows.append({
                    'patient_id': pid,
                    'action': action,
                    'status': st.get('status', 'todo'),
                    'kept_frames': ','.join(str(f) for f in kept),
                    'n_kept': len(kept),
                    'auto_kept': ','.join(str(f) for f in auto),
                    'n_auto': len(auto),
                    'n_human_edits': len(set(kept) ^ set(auto)),  # diff from model
                    'flags': '|'.join(st.get('flags', [])),
                    'note': st.get('note', ''),
                    'curated_at': st.get('curated_at', ''),
                })
        pd.DataFrame(rows).to_csv(config.CURATED_CSV, index=False)
        return config.CURATED_CSV

    def export_long_xlsx(self):
        """Combined long-format Excel: one row per curated frame, carrying the
        source AUs (dynamic + static) + action, plus provenance columns
        curated_keep / auto_keep / human_reviewed. Returns the written path, or
        None if nothing has been curated yet."""
        out = []
        for pid, pnode in self.curation.get('patients', {}).items():
            if not isinstance(pnode, dict):
                continue
            df = self.get_frame_df(pid)
            if df is None:
                continue
            for action, anode in pnode.items():
                if not isinstance(anode, dict):
                    continue
                sub = df[df['action'] == action].copy()
                if sub.empty:
                    continue
                kept = set(anode.get('kept', []))
                auto = anode.get('auto_kept')
                auto = (set(auto) if auto is not None
                        else set(self.auto_keep_frames(pid, action)))
                reviewed = bool(anode.get('status') == 'done')
                flags = set(anode.get('flags', []))
                sub.insert(0, 'patient_id', pid)
                sub['auto_keep'] = sub['frame'].isin(auto)
                sub['curated_keep'] = sub['frame'].isin(kept)
                sub['human_reviewed'] = reviewed
                sub['did_not_perform'] = 'not_performed' in flags
                sub['abnormal'] = 'abnormal' in flags
                sub['note'] = anode.get('note', '')
                out.append(sub)
        if not out:
            return None
        big = pd.concat(out, ignore_index=True)
        big = big.sort_values(['patient_id', 'frame']).reset_index(drop=True)
        try:
            big.to_excel(config.CURATED_XLSX, index=False)
            return (config.CURATED_XLSX, len(big))
        except Exception:
            alt = str(config.CURATED_XLSX).replace('.xlsx', '.csv')
            big.to_csv(alt, index=False)
            return (alt, len(big))
