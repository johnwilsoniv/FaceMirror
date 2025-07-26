"""
Frame extraction module for facial AU analysis.
Extracts frames from videos at points of maximal expression.
V1.2 Fix: Handle missing 'max_frame' key gracefully.
V1.3 Fix: Correct path logic to prevent nested patient folders.
"""

import os
import cv2
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__) # Assuming configured by main

class FacialFrameExtractor:
    """
    Extracts frames from videos at points of maximal expression.
    """

    def __init__(self):
        """Initialize the frame extractor."""
        pass

    def extract_frames(self, analyzer, video_path, output_dir, patient_id, results, action_descriptions):
        """
        Extract frames from video at points of maximal expression
        and save them as images with appropriate labels.

        Args:
            analyzer: The analyzer instance (used for context, e.g., patient_id if needed).
            video_path (str): Path to video file.
            output_dir (str): Directory to save extracted frames (SHOULD be patient-specific).
            patient_id (str): Patient ID (used for labeling).
            results (dict): Results dictionary (should contain action results with 'max_frame').
            action_descriptions (dict): Dictionary of action descriptions.

        Returns:
            tuple: (bool, dict) Success flag and dictionary of frame paths.
        """
        if not results:
            logger.error("No analysis results available for frame extraction")
            return False, {}

        # --- FIX: Use output_dir directly ---
        # The 'output_dir' argument passed to this function IS the patient-specific directory.
        # DO NOT join patient_id again here.
        # Remove: patient_output_dir = os.path.join(output_dir, patient_id)
        # Ensure the directory exists (harmless if it does)
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
             logger.error(f"Could not create or access output directory '{output_dir}'. Error: {e}")
             return False, {}
        # --- END FIX ---

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file {video_path}")
            return False, {}

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video FPS: {fps}")
        logger.info(f"Total frames: {frame_count}")

        frame_paths = {}
        frames_extracted_count = 0

        for action, info in results.items():
             if action == 'patient_summary': continue # Skip summary info

             if not isinstance(info, dict):
                 logger.warning(f"Skipping frame extraction for action '{action}': Invalid info type ({type(info)}).")
                 continue
             frame_num_any = info.get('max_frame')
             if frame_num_any is None:
                 logger.warning(f"Skipping frame extraction for action '{action}': 'max_frame' key missing in results.")
                 continue
             try:
                 frame_num = int(frame_num_any)
                 if not (0 <= frame_num < frame_count):
                     logger.error(f"Skipping frame extraction for action '{action}': Frame number {frame_num} out of bounds (0-{frame_count-1}).")
                     continue
             except (ValueError, TypeError):
                  logger.error(f"Could not convert max_frame '{frame_num_any}' to int for action '{action}'. Skipping frame.")
                  continue

             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
             ret, frame = cap.read()

             if not ret:
                 logger.error(f"Could not read frame {frame_num} for action {action}")
                 continue

             action_desc = action_descriptions.get(action, action)
             # Use patient_id passed as argument for the label
             label = f"{patient_id}_{action_desc}_frame{frame_num}"

             labeled_frame = frame.copy()
             cv2.putText(labeled_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

             # --- FIX: Use output_dir directly for path construction ---
             output_path_jpg = os.path.join(output_dir, f"{label}.jpg")
             try:
                 cv2.imwrite(output_path_jpg, labeled_frame)
                 logger.info(f"Saved labeled frame: {output_path_jpg}")
             except Exception as e_jpg:
                 logger.error(f"Failed to save labeled JPG {output_path_jpg}: {e_jpg}")
                 continue

             output_path_png = os.path.join(output_dir, f"{label}_original.png")
             # --- END FIX ---
             try:
                 cv2.imwrite(output_path_png, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                 frame_paths[action] = output_path_png
                 frames_extracted_count += 1
             except Exception as e_png:
                 logger.error(f"Failed to save original PNG {output_path_png}: {e_png}")

        cap.release()
        logger.info(f"Attempted extraction for {len(results)-1 if 'patient_summary' in results else len(results)} actions. Successfully extracted {frames_extracted_count} frames for patient {patient_id}")
        return True, frame_paths