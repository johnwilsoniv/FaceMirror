#!/usr/bin/env python3
"""
S2.5 Frame Curator — entry point.

Interactive frame-selection tool that sits between S2 (action coding) and S3
(analysis). The researcher curates which frames within each (patient, action)
are genuine, representative task performance, producing clean input for the
characteristic-window aggregator.

Startup mirrors the rest of the pipeline: a splash screen, then a folder picker
to choose the coded-data location. Already-curated work is loaded by default
(autosaved to S25 Curated Files), so a half-finished session resumes where it
left off — the sidebar marks each patient complete / in-progress / untouched.

Run:  python main.py
"""
import sys
from pathlib import Path

from PyQt5.QtWidgets import (QApplication, QFileDialog, QSplashScreen,
                             QMessageBox)
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont
from PyQt5.QtCore import Qt

import config
from data_manager import DataManager
from curator_window import CuratorWindow


def _make_splash():
    """Native Qt splash (no tkinter dependency, unlike S1/S2's Tk splash)."""
    w, h = 480, 270
    pm = QPixmap(w, h)
    pm.fill(QColor('#ffffff'))
    p = QPainter(pm)
    p.setPen(QColor('#bdc3c7'))
    p.drawRect(0, 0, w - 1, h - 1)
    p.setPen(QColor('#2c3e50'))
    p.setFont(QFont('Helvetica', 26, QFont.Bold))
    p.drawText(0, 70, w, 50, Qt.AlignCenter, "S2.5 Frame Curator")
    p.setPen(QColor('#7f8c8d'))
    p.setFont(QFont('Helvetica', 11))
    p.drawText(0, 122, w, 24, Qt.AlignCenter, f"Version {config.VERSION}")
    p.end()
    return QSplashScreen(pm)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("S2.5 Frame Curator")
    app.setStyle("Fusion")   # match S1/S2/S3 light Fusion look

    # --- file selector (like the rest of the pipeline): pick the coded-data
    # folder. Defaults to the configured location so the common case is one Enter.
    start_dir = str(config.PER_FRAME_DIR if config.PER_FRAME_DIR.exists()
                    else Path.home())
    chosen = QFileDialog.getExistingDirectory(
        None, "Select the coded-data folder  (per-frame *_mirrored_coded.csv)",
        start_dir)
    if not chosen:
        return 0   # user cancelled the picker
    config.PER_FRAME_DIR = Path(chosen)

    # --- splash during load ---
    splash = _make_splash()
    splash.show()

    def status(msg):
        splash.showMessage(msg, Qt.AlignBottom | Qt.AlignHCenter,
                           QColor('#34495e'))
        app.processEvents()

    status("Loading patients…")
    dm = DataManager()
    if not dm.patients:
        splash.close()
        QMessageBox.critical(
            None, "No data",
            f"No per-frame CSVs found in:\n{config.PER_FRAME_DIR}\n\n"
            "Expected files like  <patient>_left_mirrored_coded.csv")
        return 1

    # resume awareness: summarize what's already curated (also shown per-patient
    # in the sidebar via the complete/partial markers).
    done, total = dm.overall_progress()
    partial = sum(1 for p in dm.patients if dm.patient_status(p) == 'partial')
    status(f"{total} patients · {done} complete · {partial} in progress")

    win = CuratorWindow(dm)
    win.show()
    splash.finish(win)
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
