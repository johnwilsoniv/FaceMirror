"""
S2.5 Frame Curator — single-frame thumbnail widget.

States: included (green) / excluded-with-reason (red) / available (gray).
Minimal-click model:
  - Available/excluded frame: click the image -> include
  - Included frame: hover -> reason buttons overlay appears -> click a reason
    -> excluded with that reason (1 hover + 1 click)
"""
import numpy as np
from PyQt5.QtWidgets import (QFrame, QVBoxLayout, QLabel, QWidget,
                             QPushButton, QGridLayout)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap

import config


def ndarray_to_pixmap(img):
    if img is None:
        return None
    h, w = img.shape[:2]
    arr = np.ascontiguousarray(img)
    qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class ReasonOverlay(QWidget):
    """Floating reason-button grid shown over an included thumbnail on hover."""
    reason_picked = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background-color: rgba(20,20,20,180);")
        grid = QGridLayout(self)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setSpacing(3)
        hint = QLabel("Exclude — why?")
        hint.setStyleSheet("color: white; font-size: 9px; font-weight: bold;")
        hint.setAlignment(Qt.AlignCenter)
        grid.addWidget(hint, 0, 0, 1, 3)
        for i, reason in enumerate(config.EXCLUSION_REASONS):
            btn = QPushButton(config.EXCLUSION_SHORT.get(reason, reason))
            btn.setStyleSheet(
                "QPushButton { background:#444; color:white; font-size:9px; "
                "border:1px solid #888; border-radius:3px; padding:2px; }"
                "QPushButton:hover { background:#c0392b; }")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _, r=reason: self.reason_picked.emit(r))
            grid.addWidget(btn, 1 + i // 3, i % 3)
        self.hide()


class FrameThumbnail(QFrame):
    # frame, included(bool), reason(str|None)
    toggled = pyqtSignal(int, bool, object)

    def __init__(self, frame_idx, pixmap, info_text, state, reason=None,
                 parent=None):
        super().__init__(parent)
        self.frame_idx = frame_idx
        self.state = state           # 'included' | 'excluded' | 'available'
        self.reason = reason
        self.setFixedSize(config.THUMB_W + 8,
                          config.THUMB_H + 34)
        self.setMouseTracking(True)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(1)

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setFixedSize(config.THUMB_W, config.THUMB_H)
        if pixmap is not None:
            self.img_label.setPixmap(pixmap)
        else:
            self.img_label.setText("(no frame)")
            self.img_label.setStyleSheet("color:#999; font-size:9px;")
        lay.addWidget(self.img_label, alignment=Qt.AlignHCenter)

        self.info_label = QLabel(info_text)
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet(
            "font-size: 8px; font-family: 'Menlo','Courier New',monospace; "
            "color: #333333;")
        self.info_label.setWordWrap(True)
        lay.addWidget(self.info_label)

        # Reason overlay (child of img_label so it floats over the image)
        self.overlay = ReasonOverlay(self.img_label)
        self.overlay.setGeometry(0, 0, config.THUMB_W, config.THUMB_H)
        self.overlay.reason_picked.connect(self._on_reason)

        self._apply_style()

    # ---------- styling ----------

    def _border_color(self):
        return {'included': config.COLOR_INCLUDED,
                'excluded': config.COLOR_EXCLUDED,
                'available': config.COLOR_AVAILABLE}[self.state]

    def _apply_style(self):
        w = 3 if self.state in ('included', 'excluded') else 1
        self.setStyleSheet(
            f"FrameThumbnail {{ border: {w}px solid {self._border_color()}; "
            f"border-radius: 4px; background: white; }}")
        tag = ''
        if self.state == 'excluded' and self.reason:
            tag = f'  ✗ {config.EXCLUSION_SHORT.get(self.reason, self.reason)}'
        elif self.state == 'included':
            tag = '  ✓'
        base = self.info_label.text().split('\n')[0]
        # keep AU line, append state tag to the frame line
        lines = self.info_label.text().split('\n')
        lines[0] = lines[0].split('  ')[0] + tag
        self.info_label.setText('\n'.join(lines))

    def set_state(self, state, reason=None):
        self.state = state
        self.reason = reason
        # rebuild info first line tag cleanly
        lines = self.info_label.text().split('\n')
        lines[0] = lines[0].split('  ')[0]
        self.info_label.setText('\n'.join(lines))
        self._apply_style()

    # ---------- interaction ----------

    def enterEvent(self, event):
        if self.state == 'included':
            self.overlay.show()
            self.overlay.raise_()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.overlay.hide()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        # Clicking the body of an excluded/available frame -> include.
        # (Included frames are excluded via the hover reason buttons.)
        if self.state in ('excluded', 'available'):
            self.set_state('included')
            self.toggled.emit(self.frame_idx, True, None)
        super().mousePressEvent(event)

    def _on_reason(self, reason):
        self.set_state('excluded', reason)
        self.overlay.hide()
        self.toggled.emit(self.frame_idx, False, reason)
