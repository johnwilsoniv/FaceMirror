#!/usr/bin/env python3
"""
Warning dialog for poor landmark quality detection.

Shows a warning when many frames have poor landmark quality,
suggesting corrective actions to the user.
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QTextEdit, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap, QIcon


class LandmarkQualityWarningDialog(QDialog):
    """
    Dialog warning about poor landmark quality with recommendations.

    Shows when >30% of frames have poor landmark detection quality,
    with actionable recommendations for improving results.
    """

    def __init__(self, parent=None, poor_frame_count=0, total_frames=0,
                 poor_frames_details=None):
        """
        Initialize warning dialog.

        Args:
            parent: Parent widget
            poor_frame_count: Number of frames with poor landmarks
            total_frames: Total frames analyzed
            poor_frames_details: List of (frame_num, reason) tuples
        """
        super().__init__(parent)

        self.poor_frame_count = poor_frame_count
        self.total_frames = total_frames
        self.poor_frames_details = poor_frames_details or []
        self.user_choice = None  # 'continue', 'cancel', or None

        self.setWindowTitle("Landmark Quality Warning")
        self.setModal(True)
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        self._init_ui()

    def _init_ui(self):
        """Initialize user interface."""
        layout = QVBoxLayout()
        layout.setSpacing(15)

        # Warning header
        header_layout = QHBoxLayout()

        # Warning icon
        warning_label = QLabel("⚠️")
        warning_font = QFont()
        warning_font.setPointSize(48)
        warning_label.setFont(warning_font)
        header_layout.addWidget(warning_label)

        # Warning title
        title_label = QLabel("Poor Landmark Quality Detected")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        layout.addLayout(header_layout)

        # Statistics
        poor_percentage = (self.poor_frame_count / self.total_frames * 100) if self.total_frames > 0 else 0
        stats_text = f"<b>{self.poor_frame_count} of {self.total_frames} frames ({poor_percentage:.1f}%)</b> have poor landmark detection quality."
        stats_label = QLabel(stats_text)
        stats_label.setWordWrap(True)
        layout.addWidget(stats_label)

        # Explanation
        explanation = QLabel(
            "Poor landmark quality can result in:\n"
            "• Incorrect facial midline calculation\n"
            "• Distorted mirrored face outputs\n"
            "• Inaccurate facial measurements"
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        # Common causes section
        causes_label = QLabel("<b>Common Causes:</b>")
        layout.addWidget(causes_label)

        causes_text = QTextEdit()
        causes_text.setReadOnly(True)
        causes_text.setMaximumHeight(120)

        # Analyze most common reasons
        reasons_count = {}
        for _, reason in self.poor_frames_details[:100]:  # Sample first 100
            if reason.startswith('clustering'):
                reasons_count['clustering'] = reasons_count.get('clustering', 0) + 1
            elif reason == 'poor_distribution':
                reasons_count['poor_distribution'] = reasons_count.get('poor_distribution', 0) + 1

        causes_html = "<ul style='margin: 5px;'>"
        if reasons_count.get('clustering', 0) > reasons_count.get('poor_distribution', 0):
            causes_html += "<li><b>Surgical markings or facial asymmetry</b> - Landmarks clustering on one side of face</li>"
        else:
            causes_html += "<li><b>Poor lighting or occlusions</b> - Landmarks not well distributed</li>"

        causes_html += """
        <li>Head rotation or tilted camera angle</li>
        <li>Low image quality or resolution</li>
        </ul>
        """

        causes_text.setHtml(causes_html)
        layout.addWidget(causes_text)

        # Recommendations section
        recommendations_label = QLabel("<b>Recommendations to Improve Quality:</b>")
        layout.addWidget(recommendations_label)

        recommendations_text = QTextEdit()
        recommendations_text.setReadOnly(True)
        recommendations_text.setMaximumHeight(150)
        recommendations_html = """
        <ol style='margin: 5px;'>
        <li><b>Remove surgical markings</b> before recording (most effective)</li>
        <li><b>Ensure proper lighting</b> - Even, frontal illumination</li>
        <li><b>Keep head level</b> - Face camera directly, avoid tilting</li>
        <li><b>Check focus and resolution</b> - Ensure face is clearly visible</li>
        <li><b>Avoid occlusions</b> - Remove glasses, hair from face</li>
        </ol>
        <p style='margin-top: 10px;'><b>Note:</b> PDM shape constraints are enabled and will attempt to regularize landmarks, but cannot fully compensate for severe marking or asymmetry.</p>
        """
        recommendations_text.setHtml(recommendations_html)
        layout.addWidget(recommendations_text)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel Processing")
        cancel_btn.clicked.connect(self._on_cancel)
        button_layout.addWidget(cancel_btn)

        continue_btn = QPushButton("Continue Anyway")
        continue_btn.setDefault(True)
        continue_btn.clicked.connect(self._on_continue)
        continue_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        button_layout.addWidget(continue_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _on_continue(self):
        """User chose to continue processing."""
        self.user_choice = 'continue'
        self.accept()

    def _on_cancel(self):
        """User chose to cancel processing."""
        self.user_choice = 'cancel'
        self.reject()

    def get_user_choice(self):
        """
        Get user's choice after dialog closes.

        Returns:
            str: 'continue' or 'cancel'
        """
        return self.user_choice


def show_landmark_quality_warning(poor_frame_count, total_frames, poor_frames_details=None, parent=None):
    """
    Show landmark quality warning dialog.

    Args:
        poor_frame_count: Number of frames with poor landmarks
        total_frames: Total frames analyzed
        poor_frames_details: List of (frame_num, reason) tuples
        parent: Parent widget

    Returns:
        str: 'continue' if user wants to continue, 'cancel' otherwise
    """
    dialog = LandmarkQualityWarningDialog(
        parent=parent,
        poor_frame_count=poor_frame_count,
        total_frames=total_frames,
        poor_frames_details=poor_frames_details
    )

    result = dialog.exec_()

    if result == QDialog.Accepted:
        return 'continue'
    else:
        return 'cancel'


# Test function
if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)

    # Test with sample data
    poor_frames = [(i, 'clustering_0.76' if i % 2 == 0 else 'poor_distribution')
                   for i in range(100)]

    choice = show_landmark_quality_warning(
        poor_frame_count=350,
        total_frames=901,
        poor_frames_details=poor_frames
    )

    print(f"User choice: {choice}")
    sys.exit(0)
