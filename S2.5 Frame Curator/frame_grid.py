"""
S2.5 Frame Curator — scrollable thumbnail grid.
"""
from PyQt5.QtWidgets import QScrollArea, QWidget, QGridLayout
from PyQt5.QtCore import Qt, pyqtSignal

import config
from frame_thumbnail import FrameThumbnail, ndarray_to_pixmap


class FrameGrid(QScrollArea):
    frame_toggled = pyqtSignal(int, bool, object)  # frame, included, reason

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("QScrollArea { background: white; border: 1px solid #cccccc; }")
        self._inner = QWidget()
        self._inner.setStyleSheet("background: white;")
        self._grid = QGridLayout(self._inner)
        self._grid.setSpacing(config.THUMB_PAD)
        self._grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setWidget(self._inner)
        self._thumbs = {}

    def clear(self):
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        self._thumbs = {}

    def populate(self, frame_specs):
        """frame_specs: list of dicts {frame, pixmap, info, state, reason}."""
        self.clear()
        cols = max(1, (self.viewport().width()) //
                   (config.THUMB_W + 8 + config.THUMB_PAD))
        for i, spec in enumerate(frame_specs):
            thumb = FrameThumbnail(
                spec['frame'], spec['pixmap'], spec['info'],
                spec['state'], spec.get('reason'))
            thumb.toggled.connect(self.frame_toggled.emit)
            r, c = divmod(i, cols)
            self._grid.addWidget(thumb, r, c)
            self._thumbs[spec['frame']] = thumb

    def viewport_cols(self):
        return max(1, (self.viewport().width()) //
                   (config.THUMB_W + 8 + config.THUMB_PAD))
