"""
S2.5 Frame Curator — main window.

Layout:
  Left column : patient list (batch navigation).
  Main column : big filename header · ordered action-map strip · readable clip
                timeline · count cards · per-range WRAPPING frame grids · toolbar.
Single rejection axis (keep / not-characteristic); every frame starts on the
model's plateau prediction; keyboard-first; undo.
"""
from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QLabel, QListWidget, QListWidgetItem, QPushButton,
                             QFrame, QScrollArea, QApplication, QShortcut,
                             QMessageBox)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import (QColor, QKeySequence, QPainter, QBrush, QFont,
                         QPixmap, QIcon, QPen)

import os
import subprocess
import config
from data_manager import DataManager
from range_section import RangeSection
from frame_cell import FrameCell
try:
    import s2_handoff
except Exception:
    s2_handoff = None

T = config.THEME


def contrast_text(hexcol):
    """Black or white, whichever is more readable on the given hex color."""
    h = hexcol.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return '#000000' if lum > 140 else '#ffffff'


def dot_icon(color):
    """Small filled circle icon for patient-list status."""
    pm = QPixmap(14, 14)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color))
    p.drawEllipse(2, 2, 10, 10)
    p.end()
    return QIcon(pm)


class ClipTimeline(QWidget):
    """Whole-clip timeline, TRUE TO SCALE: every action window sits at its real
    position in clip time and the gaps between windows are real dead time. EVERY
    window shows its own per-frame selection — kept frames in the action's color,
    deselected frames light gray. The ACTIVE window is widened (to >=100px) so its
    defrag is legible, gets a bright-gold border, and is the only window that also
    shows amber 'review' (low-confidence) frames. Click any window to activate it."""
    action_clicked = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(26)   # a little headroom so the active window can grow taller
        self.setCursor(Qt.PointingHandCursor)
        self.span = (0.0, 1.0)
        self.ranges = []
        self.current_action = None
        self.status = {}          # {frame: 'keep'|'reject'} across ALL windows
        self.flagged = set()      # frames shown amber (active window only)
        self._hit = []            # [(x0, x1, action)] for click hit-testing

    def set_data(self, span, ranges, current_action):
        self.span = span if span[1] > span[0] else (0.0, 1.0)
        self.ranges = ranges
        self.current_action = current_action
        self.update()

    def set_status(self, status_by_frame, flagged=None):
        self.status = status_by_frame
        self.flagged = flagged or set()
        self.update()

    def mousePressEvent(self, event):
        x = event.x()
        for x0, x1, action in self._hit:
            if x0 <= x <= x1:
                self.action_clicked.emit(action)
                return

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        ty, th = 3, 20   # normal block row (active window grows above/below this)
        PAD = 2
        self._hit = []
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(T['timeline_bg'])))
        p.drawRoundedRect(0, ty, w, th, 4, 4)

        blocks = sorted(self.ranges, key=lambda b: b['start_t'])
        n = len(blocks)
        if n == 0:
            p.end(); return

        # ---- true-to-scale layout: real positions + real gaps between windows.
        # Then the active window is dilated to >=ACTIVE_W and the extra is absorbed
        # by compressing the gaps and the other windows uniformly, so everything
        # else keeps its relative scale. ----
        ACTIVE_W = 100.0
        avail = max(1.0, w - 2 * PAD)
        t_start = min(b['start_t'] for b in blocks)
        t_end = max(b['end_t'] for b in blocks)
        span_dur = max(t_end - t_start, 1e-6)
        active = set(i for i, b in enumerate(blocks)
                     if b['action'] == self.current_action and b.get('frames'))

        items = []           # alternating gaps + blocks, in clip order
        prev_end = t_start
        for i, b in enumerate(blocks):
            g = b['start_t'] - prev_end
            if g > 1e-9:
                items.append({'type': 'gap', 'nat': g / span_dur * avail})
            bwn = max(1e-6, b['end_t'] - b['start_t']) / span_dur * avail
            items.append({'type': 'block', 'idx': i, 'nat': bwn})
            prev_end = max(prev_end, b['end_t'])
        for it in items:
            it['w'] = it['nat']

        def is_active_item(it):
            return it['type'] == 'block' and it['idx'] in active
        bump = 0.0
        for it in items:
            if is_active_item(it) and it['nat'] < ACTIVE_W:
                bump += ACTIVE_W - it['nat']
                it['w'] = ACTIVE_W
        other_total = sum(it['nat'] for it in items if not is_active_item(it))
        if bump > 0 and other_total > 0:
            sc = max(0.0, (other_total - bump) / other_total)
            for it in items:
                if not is_active_item(it):
                    floor = 0.0 if it['type'] == 'gap' else 2.0
                    it['w'] = max(floor, it['nat'] * sc)
        ssum = sum(it['w'] for it in items) or 1.0
        if ssum > avail:
            k = avail / ssum
            for it in items:
                it['w'] *= k

        # ---- draw ----
        x = float(PAD)
        for it in items:
            if it['type'] == 'gap':
                x += it['w']
                continue
            b = blocks[it['idx']]
            is_cur = it['idx'] in active
            x0 = int(round(x))
            bwi = max(2, int(round(it['w'])))
            # the active window grows slightly TALLER (extends above & below the row)
            # so its size also emphasizes it; other windows keep the normal height.
            ty_b, th_b = (ty - 2, th + 4) if is_cur else (ty, th)
            act_col = config.ACTION_COLORS.get(b['action'], '#90a4ae')
            frames = b.get('frames') or []
            nf = len(frames)
            p.save()
            p.setClipRect(x0, ty_b, bwi + 1, th_b)
            if nf:
                for j, f in enumerate(frames):
                    sx = x0 + (j / nf) * bwi
                    sw = (bwi / nf) + 1.0
                    if is_cur and f in self.flagged:
                        col = config.DEFRAG_FLAG
                    elif self.status.get(f, 'reject') == 'keep':
                        col = act_col
                    else:
                        col = config.DEFRAG_REJECT
                    p.setPen(Qt.NoPen)
                    p.setBrush(QBrush(QColor(col)))
                    p.drawRect(int(sx), ty_b, int(sw), th_b)
            else:
                p.setPen(Qt.NoPen); p.setBrush(QBrush(QColor(act_col)))
                p.drawRect(x0, ty_b, bwi, th_b)
            p.restore()

            p.setBrush(Qt.NoBrush)
            if is_cur:
                # 2px gold outline drawn flush (NOT inset) so it pops out over the
                # adjacent windows; with the extra height the active window stands out
                # clearly without being as heavy as the 3px stroke.
                p.setPen(QPen(QColor(config.ACTIVE_GOLD), 2))
                p.drawRoundedRect(x0, ty_b, bwi, th_b, 3, 3)
            else:
                p.setPen(QPen(QColor('#b4b0a6'), 1))
                p.drawRoundedRect(x0, ty, bwi, th, 3, 3)

            if is_cur or bwi >= 20:
                p.setPen(QColor(contrast_text(act_col)))
                p.setFont(QFont("Arial", 9 if is_cur else 8,
                                QFont.Bold if is_cur else QFont.Normal))
                p.drawText(x0, ty, bwi, th, Qt.AlignCenter,
                           b['label'] if is_cur else b['action'])

            self._hit.append((x0, x0 + bwi, b['action']))
            x += it['w']
        p.end()


class CountCard(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 8, 14, 8)
        self.title = QLabel(title)
        self.title.setStyleSheet(f"font-size:11px; color:{T['text_muted']};")
        self.value = QLabel("0")
        self.value.setStyleSheet(
            f"font-size:22px; font-weight:bold; color:{T['text']};")
        lay.addWidget(self.title)
        lay.addWidget(self.value)
        self._style(False)

    def _style(self, review):
        bg = T['review_card'] if review else T['card_bg']
        self.setStyleSheet(
            f"CountCard {{ background:{bg}; border:1px solid {T['card_border']}; "
            f"border-radius:8px; }}")

    def set_value(self, v, review=False):
        self.value.setText(str(v))
        col = T['review_text'] if review else T['text']
        self.value.setStyleSheet(
            f"font-size:22px; font-weight:bold; color:{col};")
        self._style(review)


class _PrefetchWorker(QThread):
    """Decode+cache an action's thumbnails off the UI thread. Reuses the
    DataManager's preload (which opens its own VideoCapture and writes into the
    GIL-atomic thumb cache), so the UI thread later gets cache hits."""
    def __init__(self, dm, pid, action):
        super().__init__()
        self.dm = dm; self.pid = pid; self.action = action
    def run(self):
        try:
            self.dm.preload_action_thumbnails(self.pid, self.action)
        except Exception:
            pass


class CuratorWindow(QMainWindow):
    def __init__(self, dm: DataManager):
        super().__init__()
        self.dm = dm
        self.cur_pid = None
        self.cur_action = None
        self.sections = {}
        self.undo_stack = []
        self.redo_stack = []
        self._paint_target = None     # state applied during a drag stroke
        self._paint_snap = None       # snapshot captured at stroke start
        self._paint_anchor = None     # stroke start frame (for shift range-fill)
        self._prefetch = None         # background thumbnail prefetch thread
        self._action_dirty = False    # current action edited since load?
        self._last_paint_gpos = None  # cursor pos during an active paint stroke
        self._edge_dir = 0            # -1 up / +1 down auto-scroll during paint
        self._edge_timer = QTimer(self)
        self._edge_timer.setInterval(30)
        self._edge_timer.timeout.connect(self._edge_scroll_tick)
        # Autosave state machine: an edit (re)starts a 3s debounce; on fire we
        # write to disk and show "saved ✓" for ~1.2s, then revert to idle. The
        # in-memory state is always current, and navigation/close flush at once,
        # so the debounce never risks losing data on a normal exit.
        self._save_debounce = QTimer(self)
        self._save_debounce.setSingleShot(True)
        self._save_debounce.setInterval(3000)
        self._save_debounce.timeout.connect(lambda: self._flush_save(announce=True))
        self._save_revert = QTimer(self)
        self._save_revert.setSingleShot(True)
        self._save_revert.setInterval(1200)
        self._save_revert.timeout.connect(self._save_idle)
        self.setWindowTitle("S2.5 Frame Curator")
        # Open at our preferred size but never larger than the screen (logical
        # points), so it fits laptops at default scaling (e.g. 1440x900).
        _av = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1560, _av.width()), min(980, _av.height()))
        self._build_ui()
        self._install_shortcuts()
        if self.dm.patients:
            self.load_patient(self.dm.patients[0])

    # ---------- UI ----------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        central.setStyleSheet(
            f"QWidget {{ background:{T['app_bg']}; color:{T['text']}; font-family: Arial; }}"
            f"QListWidget {{ background:{T['card_bg']}; border:1px solid "
            f"{T['card_border']}; border-radius:6px; }}")
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ---- Left: patient list ----
        side = QWidget(); side.setFixedWidth(210)
        sl = QVBoxLayout(side); sl.setContentsMargins(0, 0, 0, 0); sl.setSpacing(5)
        self.patient_cap = QLabel("Patients")
        self.patient_cap.setStyleSheet(f"font-weight:bold; font-size:12px; color:{T['text']};")
        sl.addWidget(self.patient_cap)
        self.patient_list = QListWidget()
        self.patient_list.currentItemChanged.connect(self._on_patient_selected)
        sl.addWidget(self.patient_list, 1)
        root.addWidget(side)

        # ---- Main card ----
        card = QFrame()
        card.setStyleSheet(
            f"QFrame {{ background:{T['card_bg']}; border:1px solid "
            f"{T['card_border']}; border-radius:12px; }}")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(20, 12, 20, 10)
        cl.setSpacing(6)
        root.addWidget(card, 1)

        # Header: [filename + "patient N/total"]  |  [action strip] on the right
        hdr = QHBoxLayout()
        hl = QVBoxLayout(); hl.setSpacing(1)
        self.filename = QLabel()
        self.filename.setStyleSheet(
            f"font-size:22px; font-weight:bold; color:{T['text']}; "
            "border:none; background:transparent;")
        self.meta = QLabel()
        self.meta.setStyleSheet(
            f"font-size:12px; color:{T['text_muted']}; "
            "border:none; background:transparent;")
        hl.addWidget(self.filename)
        hl.addWidget(self.meta)
        hdr.addLayout(hl)
        hdr.addStretch(1)
        # counts (representative/rejected/flagged) in the header MIDDLE, vertically
        # centered so they add no height to the filename block.
        self.stat_lbl = QLabel()
        self.stat_lbl.setStyleSheet("border:none; background:transparent; font-size:13px;")
        hdr.addWidget(self.stat_lbl, alignment=Qt.AlignVCenter)
        hdr.addStretch(1)
        self.action_strip = QHBoxLayout(); self.action_strip.setSpacing(4)
        astrip_w = QWidget(); astrip_w.setLayout(self.action_strip)
        astrip_w.setStyleSheet("border:none;")
        hdr.addWidget(astrip_w, alignment=Qt.AlignTop | Qt.AlignRight)
        cl.addLayout(hdr)

        # Clip timeline + per-frame defrag for the current action
        self.timeline = ClipTimeline()
        self.timeline.action_clicked.connect(self._on_timeline_click)
        cl.addWidget(self.timeline)

        # Frame sections (scroll)
        self.sections_scroll = QScrollArea()
        self.sections_scroll.setWidgetResizable(True)
        self.sections_scroll.setFrameShape(QFrame.NoFrame)
        self.sections_scroll.setStyleSheet("background:transparent;")
        self.sections_inner = QWidget()
        self.sections_layout = QVBoxLayout(self.sections_inner)
        self.sections_layout.setAlignment(Qt.AlignTop)
        self.sections_layout.setSpacing(6)
        self.sections_scroll.setWidget(self.sections_inner)
        cl.addWidget(self.sections_scroll, 1)

        # Toolbar — buttons only (counts moved to the header). Uniform height +
        # consistent spacing throughout; order: undo redo | flags | <stretch> |
        # autosave  export  re-score  Confirm&Next  close.
        BTN_GAP = 8
        tb = QHBoxLayout()
        tb.setSpacing(BTN_GAP)
        flat = (f"QPushButton {{ background:{T['app_bg']}; color:{T['text']}; "
                f"border:1px solid {T['card_border']}; border-radius:6px; "
                "padding:4px 12px; font-size:12px; } "
                f"QPushButton:hover {{ background:#ece9e1; }}")
        self.btn_undo = QPushButton("↩ undo")
        self.btn_redo = QPushButton("↪ redo")
        self.btn_undo.setToolTip("Undo  ⌘Z"); self.btn_redo.setToolTip("Redo  ⌘⇧Z")
        for b in (self.btn_undo, self.btn_redo):
            b.setCursor(Qt.PointingHandCursor); b.setStyleSheet(flat)
        self.btn_undo.clicked.connect(self._undo)
        self.btn_redo.clicked.connect(self._redo)
        tb.addWidget(self.btn_undo); tb.addWidget(self.btn_redo)

        # Task-performance flag buttons (per action)
        self.btn_flag_np = QPushButton("⚑ Not Performed")
        self.btn_flag_np.setToolTip("Current action: patient did not perform the task as instructed")
        self.btn_flag_ab = QPushButton("⚑ Abnormal")
        self.btn_flag_ab.setToolTip("Current action: patient performed the task in an abnormal manner")
        for b in (self.btn_flag_np, self.btn_flag_ab):
            b.setCursor(Qt.PointingHandCursor); b.setStyleSheet(self._flag_css(False))
        self.btn_flag_np.clicked.connect(lambda: self._toggle_flag('not_performed'))
        self.btn_flag_ab.clicked.connect(lambda: self._toggle_flag('abnormal'))
        tb.addWidget(self.btn_flag_np); tb.addWidget(self.btn_flag_ab)
        tb.addStretch(1)

        # Autosave status pill (clickable = force save; label shows autosave is on)
        self.save_state = QPushButton("autosave on")
        self.save_state.setCursor(Qt.PointingHandCursor)
        self.save_state.setToolTip("Autosave is on — click to save now")
        self.save_state.setStyleSheet(self._savestate_css('idle'))
        self.save_state.clicked.connect(lambda: self._flush_save(announce=True))
        tb.addWidget(self.save_state)

        # Export + re-score: lesser-used -> outline only (colored text+border,
        # neutral fill like 'close'), to de-emphasize vs the primary button.
        def _outline_css(color):
            return (f"QPushButton {{ background:{T['app_bg']}; color:{color}; "
                    f"border:1px solid {color}; border-radius:6px; "
                    "padding:4px 12px; font-size:12px; font-weight:bold; } "
                    f"QPushButton:hover {{ background:#ece9e1; }}")
        # Reset the CURRENT action's curation back to the auto-curator picks.
        # Undoable (Ctrl+Z) + snapshot-backed; flags/notes preserved.
        self.btn_reset = QPushButton("↺ reset")
        self.btn_reset.setToolTip("Reset THIS action's curation to the auto-curator "
                                  "picks (start the scoring over). Undoable; flags/notes kept.")
        self.btn_reset.setCursor(Qt.PointingHandCursor)
        self.btn_reset.setStyleSheet(_outline_css('#b05a5a'))   # muted red
        self.btn_reset.clicked.connect(self._reset_current_action)
        tb.addWidget(self.btn_reset)

        self.btn_export = QPushButton("⤓ export xlsx")
        self.btn_export.setCursor(Qt.PointingHandCursor)
        self.btn_export.setStyleSheet(_outline_css('#0e9488'))   # teal text
        self.btn_export.clicked.connect(self._export_xlsx)
        # (added to the bar just left of Confirm & Next — see below)

        self.btn_rescore = QPushButton("↻ re-score")
        self.btn_rescore.setToolTip("Re-open this patient's video in S2 for "
                                    "re-scoring; returns here when you save & exit S2")
        self.btn_rescore.setCursor(Qt.PointingHandCursor)
        self.btn_rescore.setStyleSheet(_outline_css('#b8762e'))  # amber text
        self.btn_rescore.clicked.connect(self._rescore_in_s2)
        tb.addWidget(self.btn_rescore)
        tb.addWidget(self.btn_export)   # export sits just left of Confirm & Next

        # Primary: confirm the curated selection + advance
        self.btn_save = QPushButton("Confirm && Next")
        self.btn_save.setCursor(Qt.PointingHandCursor)
        self.btn_save.setStyleSheet(
            f"QPushButton {{ background:{T['primary']}; color:white; border:none; "
            "border-radius:7px; padding:5px 18px; font-size:13px; font-weight:bold; } "
            f"QPushButton:hover {{ background:{T['primary_hover']}; }}")
        self.btn_save.clicked.connect(self._save_next)
        tb.addWidget(self.btn_save)

        # Uniform height for every control in the toolbar.
        for b in (self.btn_undo, self.btn_redo, self.btn_flag_np, self.btn_flag_ab,
                  self.save_state, self.btn_reset, self.btn_rescore, self.btn_export,
                  self.btn_save):
            b.setFixedHeight(30)
        cl.addLayout(tb)
        # Poll for an S2 re-score response while a handoff is pending.
        self._handoff_pid = None
        self._handoff_timer = QTimer(self)
        self._handoff_timer.setInterval(1000)
        self._handoff_timer.timeout.connect(self._poll_s2_handoff)

        self._refresh_patient_list()

    def _install_shortcuts(self):
        QShortcut(QKeySequence.Undo, self, self._undo)
        QShortcut(QKeySequence.Redo, self, self._redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self._redo)
        QShortcut(QKeySequence("Return"), self, self._save_next)
        QShortcut(QKeySequence("Enter"), self, self._save_next)

    # ---------- columns ----------

    def _compute_cols(self):
        w = self.sections_scroll.viewport().width()
        if w <= 0:
            w = 1150
        cell = config.THUMB_W + 4   # cell width + grid spacing
        return max(1, (w - 4) // cell)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cols = self._compute_cols()
        for sec in self.sections.values():
            sec.reflow(cols)   # reposition only; never rebuild/delete

    # ---------- patient list ----------

    def _refresh_patient_list(self):
        self.patient_list.blockSignals(True)
        self.patient_list.clear()
        self.patient_list.setIconSize(QSize(14, 14))
        for pid in self.dm.patients:
            status = self.dm.patient_status(pid)
            done, total = self.dm.patient_progress(pid)
            # one line; colored status dot via icon; normal text
            needs_merge = self.dm.needs_merge(pid)
            # resume awareness: a leading ✓ marks fully-curated patients so a
            # returning user instantly sees what's done vs left to do.
            mark = "✓ " if status == 'done' else ""
            label = f"{mark}{pid}   {done}/{total}"
            if needs_merge:
                label += "  ⮌merge"
            # (Synthesized-BL markers intentionally NOT shown in the patient list,
            # per user pref. An elevated recovered baseline is still surfaced inside
            # the curator: load_action() adds an amber note to the meta line when the
            # BL action is viewed.)
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, pid)
            # "needs merge" (re-scored, awaiting the v1316 swap) takes priority —
            # a distinct blue-violet dot that reads as 'action required, different
            # kind' vs the green(done)/orange(review)/gray(todo) states.
            if needs_merge:
                col = '#3f51b5'
            elif status == 'done':
                col = T['keep_border']
            elif status == 'partial' or pid == self.cur_pid:
                col = config.DEFRAG_FLAG
            else:
                col = '#c9c5bc'
            it.setIcon(dot_icon(col))
            it.setForeground(QColor(T['text']))
            self.patient_list.addItem(it)
        self.patient_list.blockSignals(False)
        # title shows completed/total so progress is visible at a glance
        done, total = self.dm.overall_progress()
        self.patient_cap.setText(f"Patients  ({done}/{total})")

    def _on_patient_selected(self, cur, prev):
        if cur is None:
            return
        pid = cur.data(Qt.UserRole)
        if pid and pid != self.cur_pid:
            # commit the current action (save + checkmark) before leaving it
            if self._action_dirty:
                self._commit_current()
            self.load_patient(pid)

    def _on_timeline_click(self, action):
        self._switch_action(action)

    def _switch_action(self, action):
        """Switch to another action; if the current one has unsaved selections,
        commit it first (save + checkmark) so nothing is lost on switch."""
        if not action or action == self.cur_action:
            return
        if self._action_dirty:
            self._commit_current()
        self.load_action(action)

    # ---------- action strip ----------

    def _build_action_strip(self):
        while self.action_strip.count():
            it = self.action_strip.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        actions = self.dm.actions_time_ordered(self.cur_pid)
        for i, action in enumerate(actions):
            status = self.dm.action_status(self.cur_pid, action)
            is_cur = (action == self.cur_action)
            done_mark = '✓ ' if status == 'done' else ''
            if is_cur:
                # only the active action shows its real categorical color
                bg = config.ACTION_COLORS.get(action, '#b0bec5')
                fg = contrast_text(bg)
                border = f'2px solid {config.ACTIVE_GOLD}'   # gold = active
            else:
                bg = '#ded9d0'          # inactive / not-current actions
                fg = '#6b6760'
                border = 'none'
            btn = QPushButton(f"{done_mark}{i+1}. {action}")
            btn.setFixedHeight(24)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(
                f"QPushButton {{ background:{bg}; color:{fg}; padding:2px 8px; "
                f"font-weight:bold; border-radius:5px; font-size:11px; "
                f"border:{border}; }}")
            btn.setToolTip(config.ACTION_NAMES.get(action, action) + f" — {status}")
            btn.clicked.connect(lambda _, a=action: self._switch_action(a))
            self.action_strip.addWidget(btn)

    # ---------- loading ----------

    def load_patient(self, pid):
        self.cur_pid = pid
        # PRODUCTION rule: any action whose coded frame set changed (re-score /
        # data regeneration) auto-resets for the auto-curator + re-review. Runs
        # on every patient load; persists immediately so the reset isn't lost.
        reset = self.dm.reconcile_patient(pid)
        if reset:
            self.dm.save_curation()
            print(f"[reconcile] {pid}: reset {reset} (frames changed -> re-curate)")
        actions = self.dm.actions_time_ordered(pid)
        # Start at the first action in timeline order so the natural traversal
        # matches the patient's on-screen timeline.
        self.cur_action = (actions or [None])[0]
        self._refresh_patient_list()   # recolor: this patient -> review now
        for i in range(self.patient_list.count()):
            if self.patient_list.item(i).data(Qt.UserRole) == pid:
                self.patient_list.blockSignals(True)
                self.patient_list.setCurrentRow(i)
                self.patient_list.blockSignals(False)
                break
        if self.cur_action:
            self.load_action(self.cur_action)

    def load_action(self, action):
        self.cur_action = action
        self.undo_stack = []
        self.redo_stack = []
        self._action_dirty = False
        self._build_action_strip()
        self.save_state.setText("loading…")
        QApplication.processEvents()
        self.dm.preload_action_thumbnails(self.cur_pid, action)

        while self.sections_layout.count():
            it = self.sections_layout.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.sections = {}

        st = self.dm.get_action_state(self.cur_pid, action)
        kept = set(st['kept'])
        lowconf = self.dm.low_confidence_frames(self.cur_pid, action)
        self._lowconf = lowconf
        sums = self.dm.inset_sums(self.cur_pid, action)
        ranges = self.dm.get_ranges(self.cur_pid, action)
        multi = len(ranges) > 1
        cols = self._compute_cols()

        for ri in ranges:
            ri['label'] = f"{action} {ri['idx']}" if multi else action
            ri['show_header'] = multi   # per-range header only when >1 range
            sec = RangeSection(ri)
            sec.paint_begin.connect(self._paint_begin)
            sec.paint_to.connect(self._paint_to)
            sec.paint_done.connect(self._paint_done)
            for f in ri['frames']:
                img = self.dm.get_thumbnail(self.cur_pid, f)
                sec.add_cell_spec(f, img, self.dm.frame_time(self.cur_pid, f),
                                  sums.get(f), f in kept, f in lowconf)
            sec.build()
            sec.reflow(cols)
            self.sections_layout.addWidget(sec)
            self.sections[ri['idx']] = sec

        # header (compact: filename + patient N/total + action + frame count)
        pidx = self.dm.patients.index(self.cur_pid) + 1
        n = len(self.dm.patients)
        total_frames = len(self.dm.frames_for_action(self.cur_pid, action))
        rng_note = f"  ·  {len(ranges)} ranges" if len(ranges) > 1 else ""
        self.filename.setText(self.cur_pid)
        meta_txt = (f"patient {pidx} / {n}　·　{action}　·　"
                    f"{total_frames} frames{rng_note}")
        # Baseline-quality flag lives HERE (not in the patient list, per user
        # pref): when viewing a caveated BL, surface the finding inline.
        #   elevated → no neutral rest window existed (possible resting tone)
        #   smiling  → patient smiled throughout; least-smiling frames kept
        BL_FLAG = {
            'elevated': '⚠ elevated baseline — possible resting tone',
            'smiling':  '⚠ smiling baseline — no neutral rest; least-smiling frames kept',
        }
        flag_msg = BL_FLAG.get(self.dm.bl_quality(self.cur_pid)) if action == 'BL' else None
        if flag_msg:
            self.meta.setTextFormat(Qt.RichText)
            # wrap base in the muted color so rich text matches the plain look,
            # then append the finding in amber.
            meta_txt = (f'<span style="color:{T["text_muted"]};">{meta_txt}</span>'
                        '　·　<span style="color:#b8762e; font-weight:bold;">'
                        f'{flag_msg}</span>')
        else:
            self.meta.setTextFormat(Qt.PlainText)
        self.meta.setText(meta_txt)
        span = self.dm.clip_span(self.cur_pid, action)
        self.timeline.set_data(span, self.dm.all_action_ranges(self.cur_pid),
                               action)
        # per-frame keep/reject across ALL windows (every window shows its own
        # selection, not just the active one)
        self._win_status = {}
        for a in self.dm.actions_for_patient(self.cur_pid):
            ka = set(self.dm.get_action_state(self.cur_pid, a)['kept'])
            for fr in self.dm.frames_for_action(self.cur_pid, a):
                self._win_status[fr] = 'keep' if fr in ka else 'reject'
        self._update_counts()   # sets stat label + timeline status
        self._refresh_flag_buttons()
        self._save_idle()
        self._start_prefetch()

    # ---------- interactions ----------

    def _push_undo(self):
        self.undo_stack.append(self.dm.snapshot(self.cur_pid, self.cur_action))
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def _reset_current_action(self):
        """Reset the CURRENT action's curation to the auto-curator picks (redo the
        scoring from scratch). Undoable (Ctrl+Z) + snapshot-backed; flags + notes
        preserved. Mirrors _undo's in-place cell update — does NOT call load_action,
        which would wipe the undo stack."""
        if not self.cur_pid or not self.cur_action:
            return
        self._push_undo()                                  # Ctrl+Z restores
        self.redo_stack = []
        self.dm._snapshot_curation(f"reset_{self.cur_pid}_{self.cur_action}")
        auto = set(self.dm.reset_action_to_auto(self.cur_pid, self.cur_action))
        for sec in self.sections.values():                 # update grid in place
            for f, cell in sec.cells.items():
                cell.set_kept(f in auto)
        self._update_counts()
        self._refresh_flag_buttons()
        self._action_dirty = True
        self._schedule_save()
        self.btn_reset.setText("✓ reset to auto")
        QTimer.singleShot(2000, lambda: self.btn_reset.setText("↺ reset"))

    def _cell_for(self, frame):
        for sec in self.sections.values():
            if frame in sec.cells:
                return sec.cells[frame]
        return None

    def _ordered_frames(self):
        return sorted(f for sec in self.sections.values() for f in sec.cells)

    def _paint_begin(self, frame):
        # press flips this frame; the new state becomes the drag 'paint target'
        cell = self._cell_for(frame)
        if cell is None:
            return
        self._paint_snap = self.dm.snapshot(self.cur_pid, self.cur_action)
        self._paint_target = (not cell.kept)
        self._paint_anchor = frame
        self._apply_paint(frame)

    def _paint_to(self, gpos):
        # drag moved to gpos — paint, and auto-scroll if near a vertical edge.
        if self._paint_target is None:
            return
        self._last_paint_gpos = gpos
        self._update_edge_scroll(gpos)
        self._paint_at_gpos(gpos)

    def _paint_at_gpos(self, gpos):
        wdg = QApplication.widgetAt(gpos)
        while wdg is not None and not isinstance(wdg, FrameCell):
            wdg = wdg.parentWidget()
        if not isinstance(wdg, FrameCell):
            return
        shift = bool(QApplication.keyboardModifiers() & Qt.ShiftModifier)
        if shift and self._paint_anchor is not None:
            # range fill: apply the target to EVERY frame between the stroke
            # anchor and the hovered frame, even ones not directly hovered.
            order = self._ordered_frames()
            if self._paint_anchor in order and wdg.frame_idx in order:
                a = order.index(self._paint_anchor)
                b = order.index(wdg.frame_idx)
                lo, hi = (a, b) if a <= b else (b, a)
                for f in order[lo:hi + 1]:
                    self._apply_paint(f)
                return
        self._apply_paint(wdg.frame_idx)

    def _update_edge_scroll(self, gpos):
        """Set the auto-scroll direction based on cursor proximity to the top/
        bottom edge of the frame-gallery viewport, and run the edge timer while
        a stroke is active near an edge."""
        vp = self.sections_scroll.viewport()
        top = vp.mapToGlobal(vp.rect().topLeft()).y()
        y = gpos.y() - top
        h = vp.height()
        MARGIN = 70
        if y < MARGIN:
            self._edge_dir = -1
        elif y > h - MARGIN:
            self._edge_dir = +1
        else:
            self._edge_dir = 0
        if self._edge_dir != 0 and self._paint_target is not None:
            if not self._edge_timer.isActive():
                self._edge_timer.start()
        else:
            self._edge_timer.stop()

    def _edge_scroll_tick(self):
        if self._paint_target is None or self._edge_dir == 0:
            self._edge_timer.stop()
            return
        bar = self.sections_scroll.verticalScrollBar()
        bar.setValue(bar.value() + 22 * self._edge_dir)   # clamps at min/max
        # re-apply paint at the held cursor position so frames now scrolled
        # under it get the same keep/reject as the rest of the stroke.
        if self._last_paint_gpos is not None:
            self._paint_at_gpos(self._last_paint_gpos)

    def _apply_paint(self, frame):
        cell = self._cell_for(frame)
        if cell is None or cell.kept == self._paint_target:
            return
        self.dm.set_kept(self.cur_pid, self.cur_action, frame, self._paint_target)
        cell.set_kept(self._paint_target)
        self._action_dirty = True
        self._mark_partial(); self._update_counts()

    def _paint_done(self):
        # the whole stroke collapses into one undo step
        self._edge_timer.stop()
        self._edge_dir = 0
        self._last_paint_gpos = None
        if self._paint_target is None:
            return
        if self._paint_snap is not None:
            self.undo_stack.append(self._paint_snap)
            if len(self.undo_stack) > 50:
                self.undo_stack.pop(0)
            self.redo_stack.clear()
        self._paint_target = None
        self._paint_snap = None
        self._paint_anchor = None
        self._schedule_save()

    def _redo(self):
        if not self.redo_stack:
            return
        self.undo_stack.append(self.dm.snapshot(self.cur_pid, self.cur_action))
        snap = self.redo_stack.pop()
        self.dm.restore(self.cur_pid, self.cur_action, snap)
        kept = set(snap['kept'])
        for sec in self.sections.values():
            for f, cell in sec.cells.items():
                cell.set_kept(f in kept)
        self._mark_partial(); self._update_counts(); self._schedule_save()

    def _export_xlsx(self):
        # No pop-ups: export runs and reports non-modally on the export button.
        QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = self.dm.export_long_xlsx()
        except Exception as e:
            QApplication.restoreOverrideCursor()
            self._export_status("export failed — see console", '#c0392b')
            print(f"[export] FAILED: {e}")
            QTimer.singleShot(5000, self._export_reset_btn)
            return
        QApplication.restoreOverrideCursor()
        if result:
            path, nrows = result
            self._export_status(f"✓ exported {nrows} rows", config.DEFRAG_KEEP)
            print(f"[export] {nrows} rows -> {path}")
        else:
            self._export_status("nothing to export yet", T['text_muted'])
        QTimer.singleShot(4000, self._export_reset_btn)

    def _export_status(self, text, color='#0e9488'):
        """Non-modal export status shown on the export button itself (no pop-ups)."""
        self.btn_export.setText(text)
        self.btn_export.setStyleSheet(
            f"QPushButton {{ background:{T['app_bg']}; color:{color}; "
            f"border:1px solid {color}; border-radius:6px; "
            "padding:4px 12px; font-size:12px; font-weight:bold; } "
            f"QPushButton:hover {{ background:#ece9e1; }}")

    def _export_reset_btn(self):
        self.btn_export.setText("⤓ export xlsx")
        self.btn_export.setStyleSheet(
            f"QPushButton {{ background:{T['app_bg']}; color:#0e9488; "
            "border:1px solid #0e9488; border-radius:6px; "
            "padding:4px 12px; font-size:12px; font-weight:bold; } "
            f"QPushButton:hover {{ background:#ece9e1; }}")

    # ---------- S2 re-score handoff ----------

    def _rescore_status(self, text, color='#b8762e'):
        """Non-modal handoff status shown on the re-score button itself."""
        self.btn_rescore.setText(text)
        self.btn_rescore.setStyleSheet(
            f"QPushButton {{ background:{T['app_bg']}; color:{color}; "
            f"border:1px solid {color}; border-radius:6px; "
            "padding:4px 12px; font-size:12px; font-weight:bold; } "
            f"QPushButton:hover {{ background:#ece9e1; }}")

    def _rescore_reset_btn(self):
        self.btn_rescore.setText("↻ re-score")
        self.btn_rescore.setStyleSheet(
            f"QPushButton {{ background:{T['app_bg']}; color:#b8762e; "
            "border:1px solid #b8762e; border-radius:6px; "
            "padding:4px 12px; font-size:12px; font-weight:bold; } "
            f"QPushButton:hover {{ background:#ece9e1; }}")

    def _rescore_in_s2(self):
        """Hand the CURRENT patient to S2 for re-scoring (no pop-ups — status is
        shown on the re-score button). Writes a handoff request with the patient's
        source video, launches S2 from source, and polls for S2's response."""
        if s2_handoff is None:
            self._rescore_status("handoff unavailable", '#c0392b')
            return
        pid = self.cur_pid
        if not pid:
            return
        # already handing off? ignore repeat clicks
        if self._handoff_timer.isActive():
            return
        # S2's input = {pid}_source.* in S1O 'Combined Data' (S2 derives both
        # hemifaces + finds the paired _mirrored.csv AU files there).
        sdir = config.S1_COMBINED_DIR
        src = None
        for ext in ('.MOV', '.mov', '.mp4', '.MP4'):
            cand = sdir / f"{pid}_source{ext}"
            if cand.exists():
                src = cand
                break
        if src is None:
            self._rescore_status("no source video", '#c0392b')
            QTimer.singleShot(3000, self._rescore_reset_btn)
            return
        # S2 needs the paired _mirrored.csv AU files in Combined Data. If they're
        # not there, derive them on demand from v1316 (so any patient is re-scorable
        # without pre-generating all of them). Only block if there's no v1316 data.
        if not self.dm.ensure_mirrored_csvs(pid):
            self._rescore_status("no AU data — can't re-score", '#c0392b')
            QTimer.singleShot(4000, self._rescore_reset_btn)
            print(f"[re-score] {pid}: no mirrored CSV and no v1316 data to derive from.")
            return
        if not os.path.exists(config.S2_PYTHON):
            self._rescore_status("S2 python missing", '#c0392b')
            QTimer.singleShot(3000, self._rescore_reset_btn)
            return
        self._flush_save(announce=False)     # flush pending edits before leaving
        s2_handoff.clear_response()
        s2_handoff.write_request(pid, [str(src)], str(sdir))
        try:
            subprocess.Popen([config.S2_PYTHON, "main.py"], cwd=str(config.S2_DIR))
        except Exception as e:
            print(f"S2 launch failed: {e}")
            self._rescore_status("S2 launch failed", '#c0392b')
            QTimer.singleShot(3000, self._rescore_reset_btn)
            return
        self._handoff_pid = pid
        self._handoff_timer.start()
        self._rescore_status("⏳ re-scoring in S2…")

    def _poll_s2_handoff(self):
        if s2_handoff is None or not self._handoff_pid:
            self._handoff_timer.stop()
            return
        resp = s2_handoff.read_response()
        if not resp or resp.get('patient_id') != self._handoff_pid:
            return
        self._handoff_timer.stop()
        pid = self._handoff_pid
        self._handoff_pid = None
        status = resp.get('status')
        s2_handoff.clear_response()
        if status == 'completed':
            # PRODUCTION merge: swap the re-scored ACTION codes into v1316 (keeping
            # the validated AUs), reconcile so every changed action re-curates, and
            # reload so the timeline reflects the new coding immediately.
            ok, info = self.dm.import_rescored_actions(pid, resp.get('outputs'))
            if ok:
                self.dm.set_needs_merge(pid, False)        # action merge landed
                self.dm.save_curation()
                if pid == self.cur_pid:
                    self.load_patient(pid)                 # reconciles + rebuilds UI
                else:
                    self.dm.reconcile_patient(pid)
                    self.dm.save_curation()
                    self._refresh_patient_list()
                self._rescore_status("✓ re-scored — merged", config.DEFRAG_KEEP)
                print(f"[re-score] {pid} merged: {info}.")
            else:
                # merge refused (e.g. frame mismatch) -> leave data untouched and
                # flag for manual handling rather than silently doing nothing.
                self.dm.set_needs_merge(pid, True)
                self.dm.save_curation()
                self._refresh_patient_list()
                self._rescore_status("⚠ merge failed — see console", '#c0392b')
                print(f"[re-score] {pid} MERGE FAILED: {info}. "
                      f"outputs={resp.get('outputs')}. Flagged needs_merge.")
            QTimer.singleShot(6000, self._rescore_reset_btn)
        else:
            self._rescore_status("re-score cancelled", T['text_muted'])
            QTimer.singleShot(3000, self._rescore_reset_btn)

    # ---------- task-performance flags ----------

    # Distinct active colors so the two flags never look alike (and neither
    # clashes with the amber 'review' color used elsewhere).
    FLAG_ACTIVE_COLOR = {'not_performed': '#c0392b',   # red = didn't do the task
                         'abnormal': '#8e44ad'}        # purple = did it abnormally

    def _flag_css(self, active, flag=None):
        if active:
            col = self.FLAG_ACTIVE_COLOR.get(flag, config.DEFRAG_FLAG)
            return (f"QPushButton {{ background:{col}; color:white; "
                    "border:none; border-radius:6px; padding:4px 10px; "
                    "font-size:12px; font-weight:bold; }")
        return (f"QPushButton {{ background:{T['app_bg']}; color:{T['text_muted']}; "
                f"border:1px solid {T['card_border']}; border-radius:6px; "
                "padding:4px 10px; font-size:12px; } "
                f"QPushButton:hover {{ background:#ece9e1; }}")

    def _refresh_flag_buttons(self):
        flags = set(self.dm.get_flags(self.cur_pid, self.cur_action))
        np_on = 'not_performed' in flags
        ab_on = 'abnormal' in flags
        a = self.cur_action or ''
        self.btn_flag_np.setText(f"⚑ {a} Not Performed")
        self.btn_flag_ab.setText(f"⚑ {a} Abnormal")
        self.btn_flag_np.setStyleSheet(self._flag_css(np_on, 'not_performed'))
        self.btn_flag_ab.setStyleSheet(self._flag_css(ab_on, 'abnormal'))

    def _toggle_flag(self, flag):
        snap = self.dm.snapshot(self.cur_pid, self.cur_action)
        self.dm.toggle_flag(self.cur_pid, self.cur_action, flag)
        self.undo_stack.append(snap)
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self._action_dirty = True
        self._mark_partial()
        self._refresh_flag_buttons()
        self._refresh_patient_list()
        self._schedule_save()

    # ---------- saving (debounced autosave state machine) ----------

    def _savestate_css(self, state='idle'):
        pill = ("font-size:11px; border-radius:5px; padding:3px 9px; "
                "font-weight:bold;")
        if state == 'pending':       # saving… (amber)
            return (f"QPushButton {{ {pill} color:{T['review_text']}; "
                    f"background:{T['review_card']}; border:1px solid #c79a4e; }}")
        if state == 'saved':         # saved ✓ (solid green)
            return (f"QPushButton {{ {pill} color:white; "
                    f"background:{T['keep_border']}; border:1px solid {T['keep_border']}; }}")
        # idle: autosave on (soft green)
        return (f"QPushButton {{ {pill} color:{T['keep_border']}; "
                f"background:{T['keep_fill']}; border:1px solid {T['keep_border']}; }} "
                "QPushButton:hover { background:#dce9d2; }")

    def _schedule_save(self):
        """An edit happened: (re)start the 3s debounce and show 'saving…'.
        Rapid edits coalesce into a single disk write when the user pauses."""
        self._save_revert.stop()
        self.save_state.setText("saving…")
        self.save_state.setStyleSheet(self._savestate_css('pending'))
        self._save_debounce.start()   # restarts the 3s countdown

    def _flush_save(self, announce=True):
        """Write to disk now. announce=True → show 'saved ✓' then revert to idle
        after ~1.2s (debounce fire / manual click). announce=False → silent flush
        used on navigation/close (the next view resets the pill anyway)."""
        self._save_debounce.stop()
        self.dm.save_curation(); self.dm.export_csv()
        if announce:
            self.save_state.setText("saved ✓")
            self.save_state.setStyleSheet(self._savestate_css('saved'))
            self._save_revert.start()
        else:
            self._save_idle()

    def _save_idle(self):
        self._save_revert.stop()
        self.save_state.setText("autosave on")
        self.save_state.setStyleSheet(self._savestate_css('idle'))

    def _commit_current(self):
        """Approve the current action: mark it reviewed (done) + flush. Triggered
        when the user switches away from an action they edited (switching = the
        user has approved their selections)."""
        if not self.cur_pid or not self.cur_action:
            return
        self.dm.mark_status(self.cur_pid, self.cur_action, 'done')
        self._action_dirty = False
        self._flush_save(announce=False)
        self._build_action_strip()   # show the ✓ on the now-done action
        self._refresh_patient_list()
        self._update_save_button()

    def _start_prefetch(self):
        """Warm the NEXT action's thumbnail cache on a background thread so the
        next 'next action' is instant. One worker at a time; it reuses the
        DataManager's sequential decoder (its own VideoCapture)."""
        nxt = self._next_target()
        if nxt is None:
            return
        npid, nact = nxt
        if self._prefetch is not None and self._prefetch.isRunning():
            return
        try:
            self.dm.get_crop_box(npid)   # cache face box on main thread first
        except Exception:
            pass
        self._prefetch = _PrefetchWorker(self.dm, npid, nact)
        self._prefetch.start()

    def _undo(self):
        if not self.undo_stack:
            return
        self.redo_stack.append(self.dm.snapshot(self.cur_pid, self.cur_action))
        snap = self.undo_stack.pop()
        self.dm.restore(self.cur_pid, self.cur_action, snap)
        kept = set(snap['kept'])
        for sec in self.sections.values():
            for f, cell in sec.cells.items():
                cell.set_kept(f in kept)
        self._update_counts(); self._schedule_save()

    def _mark_partial(self):
        st = self.dm.get_action_state(self.cur_pid, self.cur_action)
        if st.get('status') == 'todo':
            st['status'] = 'partial'

    def _current_status(self):
        """Per-frame status for the current action's defrag strip."""
        st = self.dm.get_action_state(self.cur_pid, self.cur_action)
        kept = set(st['kept'])
        conf = set(st['confirmed'])
        lowconf = getattr(self, '_lowconf', set())
        status = {}
        for f in self.dm.frames_for_action(self.cur_pid, self.cur_action):
            if f in lowconf and f not in conf:
                status[f] = 'flag'
            elif f in kept:
                status[f] = 'keep'
            else:
                status[f] = 'reject'
        return status

    def _update_counts(self):
        k, rj, tr = self.dm.counts(self.cur_pid, self.cur_action)
        mut, keep_c, rev_c = T['text_muted'], T['keep_border'], T['review_text']
        flagged = (f"<b style='color:{rev_c}'>{tr} left</b>" if tr
                   else f"<b style='color:{mut}'>0</b>")
        self.stat_lbl.setText(
            f"<span style='color:{mut}'>representative</span> "
            f"<b style='color:{keep_c}'>{k}</b>"
            f"<span style='color:{mut}'>　·　rejected</span> "
            f"<b style='color:{T['text']}'>{rj}</b>"
            f"<span style='color:{mut}'>　·　flagged</span> {flagged}")
        if not hasattr(self, '_win_status'):
            self._win_status = {}
        st = self.dm.get_action_state(self.cur_pid, self.cur_action)
        kept = set(st['kept'])
        for f in self.dm.frames_for_action(self.cur_pid, self.cur_action):
            self._win_status[f] = 'keep' if f in kept else 'reject'
        flagged_set = (set(getattr(self, '_lowconf', set()))
                       - set(st.get('confirmed', [])))
        self.timeline.set_status(self._win_status, flagged_set)
        self._update_save_button()


    def _next_target(self):
        """The (pid, action) to advance to after the current one, or None at the
        very end of the batch (last action of the last patient)."""
        actions = self.dm.actions_time_ordered(self.cur_pid)
        if self.cur_action in actions:
            idx = actions.index(self.cur_action)
            if idx + 1 < len(actions):
                return (self.cur_pid, actions[idx + 1])
        if self.cur_pid in self.dm.patients:
            pidx = self.dm.patients.index(self.cur_pid)
            if pidx + 1 < len(self.dm.patients):
                npid = self.dm.patients[pidx + 1]
                nacts = self.dm.actions_time_ordered(npid)
                if nacts:
                    return (npid, nacts[0])
        return None

    def _update_save_button(self):
        """Confirm the curated selection + advance. At the very end of the batch
        it confirms + exports (Enter still triggers it)."""
        nxt = self._next_target()
        if nxt is None:
            self.btn_save.setText("Confirm && Finish")
            bg, hov = T['keep_border'], '#27572f'   # green = finish / export
        else:
            self.btn_save.setText("Confirm && Next")
            bg, hov = T['primary'], T['primary_hover']
        self.btn_save.setStyleSheet(
            f"QPushButton {{ background:{bg}; color:white; border:none; "
            "border-radius:7px; padding:5px 18px; font-size:13px; font-weight:bold; } "
            f"QPushButton:hover {{ background:{hov}; }}")

    def _save_next(self):
        nxt = self._next_target()
        if nxt is None:
            self._finish_and_export()     # terminal: guard -> export -> close
            return
        self.dm.mark_status(self.cur_pid, self.cur_action, 'done')
        self._action_dirty = False
        self._flush_save(announce=False); self._refresh_patient_list()
        npid, nact = nxt
        if npid != self.cur_pid:
            self.load_patient(npid)
        else:
            self.load_action(nact)

    def _patients_not_done(self):
        """(n_not_fully_done, n_total) over the loaded series, EXCLUDING the current
        patient (which the finish is about to confirm)."""
        total = len(self.dm.patients)
        nd = sum(1 for p in self.dm.patients
                 if p != self.cur_pid and self.dm.patient_status(p) != 'done')
        return nd, total

    def _finish_and_export(self):
        """Confirm-and-Finish (last action of the last patient): guard against a
        premature finish, export the curation, then confirm + close. Pop-ups here
        are intentional — this is the terminal step (unlike the routine export
        button, which stays non-modal)."""
        # 1) premature-finish guard: warn if other patients aren't fully reviewed
        not_done, total = self._patients_not_done()
        if not_done > 0:
            if QMessageBox.question(
                    self, "Finish curation?",
                    f"{not_done} of {total} patients in this folder are not fully "
                    "reviewed yet.\n\nFinish, export, and close anyway?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return   # abort — nothing confirmed, nothing exported
        # 2) confirm the current (last) action, then export
        self.dm.mark_status(self.cur_pid, self.cur_action, 'done')
        self._action_dirty = False
        self._flush_save(announce=False); self._refresh_patient_list()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = self.dm.export_long_xlsx()
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Export failed", str(e))
            self._update_save_button(); return
        QApplication.restoreOverrideCursor()
        if not result:
            QMessageBox.warning(self, "Export", "Nothing curated to export yet.")
            self._update_save_button(); return
        # 3) confirm export + close on OK (Cancel keeps the curator open)
        path, nrows = result
        if QMessageBox.information(
                self, "Curation exported",
                f"Exported {nrows} rows to:\n{path}\n\nClick OK to close the curator.",
                QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Ok) == QMessageBox.Ok:
            self.close()
        else:
            self._update_save_button()

    def closeEvent(self, event):
        self._flush_save(announce=False)
        if self._prefetch is not None and self._prefetch.isRunning():
            self._prefetch.wait(2000)
        super().closeEvent(event)
