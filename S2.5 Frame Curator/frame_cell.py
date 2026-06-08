"""
S2.5 Frame Curator — single frame cell.

Single rejection axis: KEPT (representative) vs rejected (not characteristic).
State is shown by BRIGHTNESS — kept frames are full-color, rejected frames are
dimmed. Low-confidence frames carry a 1px review-colored frame border.

Interaction is CLICK-AND-DRAG paint: pressing a frame flips it (keep<->reject)
and the parent records that as the 'paint target'; dragging across other frames
applies the same target to each. A plain click is just a zero-length drag, so it
toggles a single frame. The cell only emits intent (begin / move / done); the
parent owns all state so a whole stroke is one undo step.
"""
import numpy as np
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, pyqtSignal, QRectF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPainterPath

import config

T = config.THEME
CORNER_RADIUS = 7


def ndarray_to_pixmap(img, radius=CORNER_RADIUS, bg=None):
    """Convert an RGB ndarray to a QPixmap with rounded corners; the rounded-off
    corners are filled with `bg` (card background) so any dark/blue video corners
    are replaced by clean background instead of hard wedges."""
    if img is None:
        return None
    h, w = img.shape[:2]
    arr = np.ascontiguousarray(img)
    qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
    src = QPixmap.fromImage(qimg.copy())
    out = QPixmap(w, h)
    out.fill(QColor(bg or T['card_bg']))
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing)
    path = QPainterPath()
    path.addRoundedRect(QRectF(0, 0, w, h), radius, radius)
    p.setClipPath(path)
    p.drawPixmap(0, 0, src)
    p.end()
    return out


class FrameCell(QFrame):
    paint_begin = pyqtSignal(int)      # frame_idx — left press starts a stroke
    paint_to = pyqtSignal(object)      # global QPoint — drag moved here
    paint_done = pyqtSignal()          # left release ends the stroke

    def __init__(self, frame_idx, pixmap, timestamp, inset_val, kept, low_conf,
                 parent=None):
        super().__init__(parent)
        self.frame_idx = frame_idx
        self.kept = kept
        self.low_conf = low_conf
        self.setFixedSize(config.THUMB_W, config.THUMB_H)
        self.setCursor(Qt.PointingHandCursor)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setFixedSize(config.THUMB_W, config.THUMB_H)
        self.img_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        if pixmap is not None:
            self.img_label.setPixmap(pixmap)
        else:
            self.img_label.setText("(no frame)")
            self.img_label.setStyleSheet("color:#bbb; font-size:9px; border:none;")
        lay.addWidget(self.img_label, alignment=Qt.AlignHCenter)

        # Dim overlay (shown when rejected) — covers the image, rounded to match.
        self.dim = QLabel(self.img_label)
        self.dim.setFixedSize(config.THUMB_W, config.THUMB_H)
        self.dim.move(0, 0)
        self.dim.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.dim.setStyleSheet(
            "background: rgba(244,242,236,82); border:none; "
            f"border-radius:{CORNER_RADIUS}px;")

        # Low-confidence "needs review" indicator: a crisp 1px frame in the
        # review color hugging the image edge. Stacked ABOVE the dim overlay so it
        # stays visible whether the frame is kept or rejected.
        self.review_border = QLabel(self.img_label)
        self.review_border.setFixedSize(config.THUMB_W, config.THUMB_H)
        self.review_border.move(0, 0)
        self.review_border.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.review_border.setStyleSheet(
            f"border:1px solid {config.DEFRAG_FLAG}; "
            f"border-radius:{CORNER_RADIUS}px; background:transparent;")
        self.review_border.setVisible(low_conf)

        self._apply()

    def _apply(self):
        # kept = full color; rejected = dimmed. Review border stays on top.
        self.dim.setVisible(not self.kept)
        if self.low_conf:
            self.review_border.raise_()
        self.setStyleSheet("FrameCell { border:none; background:transparent; }")

    def set_kept(self, kept):
        self.kept = kept
        self._apply()

    # ---- click-and-drag paint (state owned by the parent) ----
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.paint_begin.emit(self.frame_idx)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.paint_to.emit(event.globalPos())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.paint_done.emit()
            event.accept()
        else:
            super().mouseReleaseEvent(event)
