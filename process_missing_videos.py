"""
Process the 12 missing videos and append to training_data_dir.
Run this from PyCharm with working directory set to 'SplitFace Open3'.
"""

import sys
from pathlib import Path

# Add package paths - adjust this to your SplitFace Open3 directory
SPLITFACE_DIR = Path(__file__).parent.resolve()
for subdir in ['pymtcnn', 'pyclnf', 'pyfaceau', 'pyfhog']:
    path = SPLITFACE_DIR / subdir
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cv2
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

# Missing videos
MISSING_VIDEOS = [
    "Patient Data/Normal Cohort/IMG_0428.MOV",
    "Patient Data/Normal Cohort/IMG_0433.MOV",
    "Patient Data/Normal Cohort/IMG_0434.MOV",
    "Patient Data/Normal Cohort/IMG_0435.MOV",
    "Patient Data/Normal Cohort/IMG_0438.MOV",
    "Patient Data/Normal Cohort/IMG_0452.MOV",
    "Patient Data/Normal Cohort/IMG_0453.MOV",
    "Patient Data/Normal Cohort/IMG_0579.MOV",
    "Patient Data/Normal Cohort/IMG_0942.MOV",
    "Patient Data/Paralysis Cohort/IMG_0592.MOV",
    "Patient Data/Paralysis Cohort/IMG_0861.MOV",
    "Patient Data/Paralysis Cohort/IMG_1366.MOV",
]

AU_NAMES = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09',
            'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23',
            'AU25', 'AU26', 'AU45']

OUTPUT_DIR = Path("Patient Data/training_data_dir")


def main():
    # Load existing annotations to get next sample ID
    existing_df = pd.read_csv(OUTPUT_DIR / "annotations.csv")
    next_id = len(existing_df)
    print(f"Starting from sample ID: {next_id}")

    # Initialize components
    print("Initializing components...")

    # MTCNN detector
    from pymtcnn import MTCNN
    detector = MTCNN()
    print(f"  MTCNN: {detector.backend_name}")

    # CLNF (uses package-relative models via symlink)
    from pyclnf import CLNF
    clnf = CLNF()
    print("  CLNF: initialized")

    # pyfaceau components
    from pyfaceau.alignment.calc_params import CalcParams
    from pyfaceau.features.pdm import PDMParser
    from pyfaceau.alignment.face_aligner import OpenFace22FaceAligner
    from pyfaceau.prediction.model_parser import OF22ModelParser
    from pyfaceau.features.triangulation import TriangulationParser
    import pyfhog

    pdm_path = "pyfaceau/weights/In-the-wild_aligned_PDM_68.txt"
    pdm_parser = PDMParser(pdm_path)
    calc_params = CalcParams(pdm_parser)
    face_aligner = OpenFace22FaceAligner(
        pdm_path,
        sim_scale=0.7,
        output_size=(112, 112)
    )
    triangulation = TriangulationParser("pyfaceau/weights/tris_68_full.txt")
    au_models = OF22ModelParser("pyfaceau/weights/AU_predictors").load_all_models(
        use_recommended=True, use_combined=True, verbose=False
    )
    print("  AU models: loaded\n")

    def extract_hog(aligned_face):
        """Extract HOG features from aligned face."""
        hog = pyfhog.extract_fhog_features(aligned_face, cell_size=8)
        hog = hog.reshape(12, 12, 31).transpose(1, 0, 2).flatten().astype(np.float32)
        return hog

    def predict_aus(hog_features, geom_features):
        """Predict AU intensities."""
        full_vector = np.concatenate([hog_features, geom_features])
        running_median = np.zeros_like(full_vector)

        au_values = []
        for au_name in AU_NAMES:
            if au_name not in au_models:
                au_values.append(0.0)
                continue

            model = au_models[au_name]
            is_dynamic = (model['model_type'] == 'dynamic')

            if is_dynamic:
                centered = full_vector - model['means'].flatten() - running_median
            else:
                centered = full_vector - model['means'].flatten()

            pred = np.dot(centered.reshape(1, -1), model['support_vectors']) + model['bias']
            au_values.append(float(np.clip(pred[0, 0], 0.0, 5.0)))

        return np.array(au_values, dtype=np.float32)

    def process_frame(frame):
        """Process a single frame."""
        # Face detection
        bboxes, mtcnn_landmarks = detector.detect(frame)
        if bboxes is None or len(bboxes) == 0:
            return None
        bbox = bboxes[0][:4].astype(np.float32)

        # CLNF landmark detection
        try:
            landmarks, info = clnf.fit(frame, bbox)
            landmarks = landmarks.astype(np.float32)
        except Exception:
            return None

        # Pose and PDM parameters
        try:
            global_params, local_params = calc_params.calc_params(landmarks)
            global_params = global_params.astype(np.float32)
            local_params = local_params.astype(np.float32)
        except Exception:
            return None

        # Face alignment
        try:
            tx, ty, rz = global_params[4], global_params[5], global_params[3]
            aligned_face = face_aligner.align_face(
                image=frame,
                landmarks_68=landmarks,
                pose_tx=tx,
                pose_ty=ty,
                p_rz=rz,
                apply_mask=True,
                triangulation=triangulation
            )
        except Exception:
            return None

        # HOG features
        hog_features = extract_hog(aligned_face)

        # Geometric features
        geom_features = pdm_parser.extract_geometric_features(local_params).astype(np.float32)

        # AU prediction
        au_intensities = predict_aus(hog_features, geom_features)

        # Convert BGR to RGB
        aligned_face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

        return {
            'image': aligned_face_rgb,
            'hog_features': hog_features,
            'landmarks': landmarks,
            'global_params': global_params,
            'local_params': local_params,
            'au_intensities': au_intensities,
            'bbox': bbox,
        }

    # Process each video
    all_new_annotations = []
    total_saved = 0
    total_failed = 0

    for video_path_str in MISSING_VIDEOS:
        video_path = Path(video_path_str)
        print(f"Processing: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  ERROR: Cannot open video")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        saved = 0
        failed = 0

        for frame_idx in tqdm(range(total_frames), desc=f"  {video_path.stem}"):
            ret, frame = cap.read()
            if not ret:
                break

            result = process_frame(frame)

            if result is None:
                failed += 1
                continue

            sample_id = f"{next_id:05d}"

            # Save files
            cv2.imwrite(str(OUTPUT_DIR / "images" / f"{sample_id}.png"),
                        cv2.cvtColor(result['image'], cv2.COLOR_RGB2BGR))
            np.save(OUTPUT_DIR / "landmarks" / f"{sample_id}.npy", result['landmarks'])
            np.save(OUTPUT_DIR / "hog_features" / f"{sample_id}.npy", result['hog_features'])

            # Build annotation row
            row = {
                'sample_id': sample_id,
                'video_name': video_path.name,
                'frame_idx': frame_idx,
                'bbox_x': result['bbox'][0],
                'bbox_y': result['bbox'][1],
                'bbox_w': result['bbox'][2],
                'bbox_h': result['bbox'][3],
                'pose_scale': result['global_params'][0],
                'pose_rx': result['global_params'][1],
                'pose_ry': result['global_params'][2],
                'pose_rz': result['global_params'][3],
                'pose_tx': result['global_params'][4],
                'pose_ty': result['global_params'][5],
            }

            for j in range(len(result['local_params'])):
                row[f'pdm_{j:02d}'] = result['local_params'][j]

            for j, au_name in enumerate(AU_NAMES):
                row[au_name] = result['au_intensities'][j]

            all_new_annotations.append(row)
            next_id += 1
            saved += 1

        cap.release()
        print(f"  => Saved: {saved}, Failed: {failed}")
        total_saved += saved
        total_failed += failed

    # Save results
    if all_new_annotations:
        new_df = pd.DataFrame(all_new_annotations)

        for col in ['video_name', 'frame_idx']:
            if col not in existing_df.columns:
                existing_df[col] = None

        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(OUTPUT_DIR / "annotations.csv", index=False)

        print(f"\n{'='*60}")
        print(f"COMPLETE!")
        print(f"New samples added: {total_saved}")
        print(f"Total samples now: {len(combined_df)}")
        print(f"Failed frames: {total_failed}")

    # Update metadata
    with open(OUTPUT_DIR / "metadata.json", 'r') as f:
        metadata = json.load(f)
    metadata['n_samples'] = next_id
    metadata['videos_added'] = MISSING_VIDEOS
    with open(OUTPUT_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Updated metadata.json")


if __name__ == "__main__":
    main()
