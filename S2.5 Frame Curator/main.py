#!/usr/bin/env python3
"""
S2.5 Frame Curator — entry point.

Interactive frame-selection tool that sits between S2 (action coding) and S3
(analysis). The researcher curates which frames within each (patient, action)
are genuine, representative task performance, producing clean input for the
characteristic-window aggregator.

Run:  python main.py
"""
import sys
from PyQt5.QtWidgets import QApplication

import config
from data_manager import DataManager
from curator_window import CuratorWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("S2.5 Frame Curator")
    app.setStyle("Fusion")   # match S1/S2/S3 light Fusion look

    dm = DataManager()
    if not dm.patients:
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(
            None, "No data",
            f"No per-frame CSVs found in:\n{config.PER_FRAME_DIR}\n\n"
            "Expected files like  <patient>_left_mirrored_coded.csv")
        return 1

    win = CuratorWindow(dm)
    win.show()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
