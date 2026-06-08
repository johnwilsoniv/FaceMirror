"""
S2.5 Frame Curator — per-range sub-section.

Header ("Range 1 · 16.4–18.2s · 7 frames") + per-range bulk actions, then a
WIDTH-FILLING WRAPPING GRID of frame cells (not a single row), so screen space
is used efficiently. Pooled frames are grouped by their originating S2 range so
provenance is visible even though the audit is one action.
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QGridLayout)
from PyQt5.QtCore import Qt, pyqtSignal

import config
from frame_cell import FrameCell, ndarray_to_pixmap

T = config.THEME


class RangeSection(QWidget):
    paint_begin = pyqtSignal(int)       # frame — stroke start
    paint_to = pyqtSignal(object)       # global QPoint — drag moved here
    paint_done = pyqtSignal()           # stroke end

    def __init__(self, range_info, parent=None):
        super().__init__(parent)
        self.range_info = range_info
        self.cells = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 6)
        lay.setSpacing(4)

        ri = range_info
        # Per-range header only when an action has multiple ranges (otherwise the
        # action + frame count lives in the top meta line).
        if ri.get('show_header', False):
            label = ri.get('label', f"Range {ri['idx']}")
            title = QLabel(
                f"<b>{label}</b> · {ri['start_t']:.1f}–{ri['end_t']:.1f}s "
                f"· {len(ri['frames'])} frames")
            title.setStyleSheet(f"font-size:12px; color:{T['text']};")
            lay.addWidget(title)

        self.grid = QGridLayout()
        self.grid.setContentsMargins(0, 2, 0, 0)
        self.grid.setSpacing(4)
        self.grid.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lay.addLayout(self.grid)
        self._specs = []
        self._order = []     # frame order
        self._cols = None

    def add_cell_spec(self, frame_idx, img, timestamp, inset_val, kept, low_conf):
        self._specs.append((frame_idx, img, timestamp, inset_val, kept, low_conf))

    def build(self):
        """Create the FrameCell widgets ONCE (not on resize)."""
        for it in list(self.cells.values()):
            it.setParent(None)
            it.deleteLater()
        self.cells = {}
        self._order = []
        for (f, img, ts, val, kept, lc) in self._specs:
            pm = ndarray_to_pixmap(img)
            cell = FrameCell(f, pm, ts, val, kept, lc)
            cell.paint_begin.connect(self.paint_begin.emit)
            cell.paint_to.connect(self.paint_to.emit)
            cell.paint_done.connect(self.paint_done.emit)
            self.cells[f] = cell
            self._order.append(f)

    def reflow(self, cols):
        """Reposition existing cells into a `cols`-wide grid. Never deletes —
        avoids the delete/re-add overlap race on resize."""
        if cols == self._cols:
            return
        self._cols = cols
        # detach all (without deleting), then re-add at new positions
        for f in self._order:
            self.grid.removeWidget(self.cells[f])
        for i, f in enumerate(self._order):
            r, c = divmod(i, cols)
            self.grid.addWidget(self.cells[f], r, c)

    def layout_cells(self, cols):
        """Convenience: build (if needed) + reflow."""
        if not self.cells:
            self.build()
        # force reflow even if cols unchanged after a fresh build
        self._cols = None
        self.reflow(cols)

    def set_cell_kept(self, frame_idx, kept):
        if frame_idx in self.cells:
            self.cells[frame_idx].set_kept(kept)

    def set_all_kept(self, kept):
        for c in self.cells.values():
            c.set_kept(kept)
