import cv2
import numpy as np
from pathlib import Path
import logging
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import dlib
import math
from scipy.spatial import Delaunay


class AdvancedFaceSplitter:
    def __init__(self, debug_mode=True):
        # Initialize face detector and predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        # Define symmetry pairs for 68-point facial landmarks
        self.SYMMETRY_PAIRS = [
            (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),  # Jaw
            (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),  # Eyebrows
            (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),  # Eyes
            (31, 35), (32, 34),  # Nose
            (48, 54), (49, 53), (50, 52), (59, 55), (58, 56), (57, 51)  # Mouth
        ]

        # Midline points (single points that lie on the symmetry line)
        self.MIDLINE_POINTS = [27, 28, 29, 30, 33, 51, 57, 8]

        self.debug_mode = debug_mode
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_face_mesh(self, frame):
        """Get facial landmarks and create a mesh"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if not faces:
            return None, None

        # Get largest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())

        # Get landmarks
        landmarks = self.predictor(gray, face)
        points = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Create mesh using Delaunay triangulation
        hull = cv2.convexHull(points)
        triangles = Delaunay(points)

        return points, triangles

    def calculate_true_midline(self, landmarks):
        """Calculate true facial midline using symmetry pairs"""
        if landmarks is None:
            return None

        # Calculate midpoints of symmetry pairs
        midpoints = []
        for pair in self.SYMMETRY_PAIRS:
            point1 = landmarks[pair[0]]
            point2 = landmarks[pair[1]]
            midpoint = (point1 + point2) / 2
            midpoints.append(midpoint)

        # Add explicit midline points
        for idx in self.MIDLINE_POINTS:
            midpoints.append(landmarks[idx])

        midpoints = np.array(midpoints)

        # Fit line to midpoints using RANSAC for robustness
        vx, vy, x0, y0 = cv2.fitLine(midpoints, cv2.DIST_HUBER, 0, 0.01, 0.01)

        return {
            'direction': (vx[0], vy[0]),
            'point': (x0[0], y0[0]),
            'midpoints': midpoints
        }

    def align_face(self, frame, landmarks, midline):
        """Align face vertically based on true midline"""
        if landmarks is None or midline is None:
            return frame, None

        # Calculate rotation angle
        angle = math.atan2(midline['direction'][1], midline['direction'][0])
        angle_deg = math.degrees(angle)

        # Get rotation matrix
        center = tuple(map(int, midline['point']))
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

        # Apply rotation
        aligned_frame = cv2.warpAffine(frame, rotation_matrix, (frame.shape[1], frame.shape[0]))

        # Transform landmarks
        aligned_landmarks = np.hstack([landmarks, np.ones((landmarks.shape[0], 1))])
        aligned_landmarks = np.dot(rotation_matrix, aligned_landmarks.T).T

        return aligned_frame, aligned_landmarks

    def create_mirrored_faces(self, frame, landmarks, midline):
        """Create left and right face images using proper warping"""
        if landmarks is None or midline is None:
            return frame, frame

        height, width = frame.shape[:2]

        # Create masks for left and right sides
        mask = np.zeros((height, width), dtype=np.float32)
        cv2.fillConvexPoly(mask, cv2.convexHull(landmarks.astype(int)), 1)

        # Split mask at midline
        x_mid = int(midline['point'][0])
        left_mask = mask.copy()
        left_mask[:, x_mid:] = 0
        right_mask = mask.copy()
        right_mask[:, :x_mid] = 0

        # Create mirrored images
        left_face = frame.copy()
        right_face = frame.copy()

        # Mirror left side
        left_half = cv2.multiply(frame, cv2.cvtColor(left_mask, cv2.COLOR_GRAY2BGR))
        mirrored_left = cv2.flip(left_half, 1)
        left_face = cv2.add(left_half, cv2.flip(left_half, 1))

        # Mirror right side
        right_half = cv2.multiply(frame, cv2.cvtColor(right_mask, cv2.COLOR_GRAY2BGR))
        mirrored_right = cv2.flip(right_half, 1)
        right_face = cv2.add(right_half, cv2.flip(right_half, 1))

        return left_face, right_face

    def visualize_debug(self, frame, landmarks, midline, triangles=None):
        """Create debug visualization showing mesh and midline"""
        debug_frame = frame.copy()

        if landmarks is not None:
            # Draw facial landmarks
            for point in landmarks.astype(int):
                cv2.circle(debug_frame, tuple(point), 2, (0, 255, 0), -1)

            # Draw symmetry pairs
            for pair in self.SYMMETRY_PAIRS:
                pt1 = tuple(landmarks[pair[0]].astype(int))
                pt2 = tuple(landmarks[pair[1]].astype(int))
                cv2.line(debug_frame, pt1, pt2, (255, 0, 0), 1)

            # Draw midline points
            for idx in self.MIDLINE_POINTS:
                point = tuple(landmarks[idx].astype(int))
                cv2.circle(debug_frame, point, 3, (0, 0, 255), -1)

            # Draw triangulation mesh if available
            if triangles is not None:
                points = landmarks.astype(np.int32)
                for simplex in triangles.simplices:
                    cv2.line(debug_frame, tuple(points[simplex[0]]), tuple(points[simplex[1]]), (0, 255, 255), 1)
                    cv2.line(debug_frame, tuple(points[simplex[1]]), tuple(points[simplex[2]]), (0, 255, 255), 1)
                    cv2.line(debug_frame, tuple(points[simplex[2]]), tuple(points[simplex[0]]), (0, 255, 255), 1)

            # Draw true midline
            if midline:
                direction = midline['direction']
                point = midline['point']
                length = 200
                pt1 = (int(point[0] - direction[0] * length), int(point[1] - direction[1] * length))
                pt2 = (int(point[0] + direction[0] * length), int(point[1] + direction[1] * length))
                cv2.line(debug_frame, pt1, pt2, (0, 255, 255), 2)

                # Draw midpoints used for midline calculation
                for point in midline['midpoints']:
                    cv2.circle(debug_frame, tuple(point.astype(int)), 2, (255, 255, 0), -1)

        return debug_frame

    def process_video(self, input_path, output_dir):
        """Process video with advanced face alignment and mirroring"""
        # Setup paths and video capture
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        left_output = output_dir / f"{input_path.stem}_left_mirrored.mp4"
        right_output = output_dir / f"{input_path.stem}_right_mirrored.mp4"
        debug_output = output_dir / f"{input_path.stem}_debug.mp4"

        self.logger.info(f"Setting up video processing:")
        self.logger.info(f"Input file: {input_path}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Debug output will be: {debug_output}")

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Error opening video file: {input_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.logger.info(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")

        # Initialize video writers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        left_writer = cv2.VideoWriter(str(left_output), fourcc, fps, (width, height))
        right_writer = cv2.VideoWriter(str(right_output), fourcc, fps, (width, height))
        debug_writer = cv2.VideoWriter(str(debug_output), fourcc, fps, (width, height))

        # Check if writers were initialized properly
        if not debug_writer.isOpened():
            self.logger.error("Failed to create debug video writer!")
        else:
            self.logger.info("Debug video writer created successfully")

        frame_count = 0
        debug_frames_written = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 30 == 0:
                self.logger.info(f"Processing frame {frame_count}/{total_frames}")

            try:
                # Get face mesh and landmarks
                landmarks, triangles = self.get_face_mesh(frame)

                # Create debug visualization
                debug_frame = self.visualize_debug(frame, landmarks,
                                                   self.calculate_true_midline(
                                                       landmarks) if landmarks is not None else None,
                                                   triangles)

                # Write debug frame
                if debug_writer.isOpened():
                    debug_writer.write(debug_frame)
                    debug_frames_written += 1

                if landmarks is not None:
                    # Calculate true midline
                    midline = self.calculate_true_midline(landmarks)

                    # Align face vertically
                    aligned_frame, aligned_landmarks = self.align_face(frame, landmarks, midline)

                    if aligned_landmarks is not None:
                        # Create mirrored faces using aligned frame
                        left_face, right_face = self.create_mirrored_faces(
                            aligned_frame, aligned_landmarks, midline
                        )
                    else:
                        left_face, right_face = frame.copy(), frame.copy()
                else:
                    left_face, right_face = frame.copy(), frame.copy()

                # Write output frames
                left_writer.write(left_face)
                right_writer.write(right_face)

            except Exception as e:
                self.logger.warning(f"Error processing frame {frame_count}: {str(e)}")
                left_writer.write(frame)
                right_writer.write(frame)
                debug_writer.write(frame)

        # Cleanup
        cap.release()
        left_writer.release()
        right_writer.release()
        debug_writer.release()

        self.logger.info(f"Processing complete:")
        self.logger.info(f"Total frames processed: {frame_count}")
        self.logger.info(f"Debug frames written: {debug_frames_written}")

        # Verify files were created
        if debug_output.exists():
            self.logger.info(f"Debug video created successfully at {debug_output}")
            self.logger.info(f"Debug video size: {debug_output.stat().st_size} bytes")
        else:
            self.logger.error(f"Failed to create debug video at {debug_output}")

        return str(left_output), str(right_output), str(debug_output)


def select_video_files():
    """Open a file dialog to select multiple video files"""
    root = tk.Tk()
    root.withdraw()

    file_paths = filedialog.askopenfilenames(
        title="Select Video Files",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
    )

    return file_paths if file_paths else None


def main():
    """GUI-based usage for multiple files"""
    try:
        # Select input video files
        input_paths = select_video_files()
        if not input_paths:
            logging.info("No video files selected. Exiting...")
            return

        # Create output directory
        script_dir = Path.cwd()  # Gets current working directory
        logging.info("Script Dir: ", script_dir)
        output_dir = script_dir / 'output'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process videos
        splitter = AdvancedFaceSplitter(debug_mode=True)
        results = []

        for input_path in input_paths:
            try:
                left_video, right_video, debug_video = splitter.process_video(input_path, output_dir)
                results.append({
                    'input': input_path,
                    'success': True,
                    'outputs': (left_video, right_video, debug_video)
                })
            except Exception as e:
                logging.error(f"Error processing {input_path}: {str(e)}")
                results.append({
                    'input': input_path,
                    'success': False,
                    'error': str(e)
                })

        # Show results
        root = tk.Tk()
        root.withdraw()

        summary = f"Output Directory: {output_dir}\n\nProcessing Results:\n\n"
        for result in results:
            if result['success']:
                summary += f"✓ {Path(result['input']).name}\n"
                summary += f"  - Left: {Path(result['outputs'][0]).name}\n"
                summary += f"  - Right: {Path(result['outputs'][1]).name}\n"
                summary += f"  - Debug: {Path(result['outputs'][2]).name}\n"
            else:
                summary += f"✗ {Path(result['input']).name} - Error: {result['error']}\n"

        messagebox.showinfo("Processing Complete", summary)

    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        error_msg = f"Error processing videos: {str(e)}"
        messagebox.showerror("Error", error_msg)
        logging.error(error_msg)


if __name__ == "__main__":
    main()