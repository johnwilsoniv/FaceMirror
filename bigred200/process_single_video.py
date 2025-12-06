"""
Process a single video - designed for SLURM array jobs
Usage: python process_single_video.py <video_path> --output-dir <output_dir>
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
import time

# Initialize pipeline components
def init_pipeline():
    from pymtcnn import MTCNN
    from pyclnf import CLNF
    from pyfaceau.alignment.calc_params import CalcParams
    from pyfaceau.features.pdm import PDMParser
    from pyfaceau.alignment.face_aligner import OpenFace22FaceAligner
    from pyfaceau.prediction.model_parser import OF22ModelParser
    from pyfaceau.features.triangulation import TriangulationParser
    import pyfhog as fhog

    print("Initializing pipeline components...")

    detector = MTCNN()
    clnf = CLNF(model_dir="pyclnf/pyclnf/models")

    pdm_path = "pyfaceau/weights/In-the-wild_aligned_PDM_68.txt"
    pdm_parser = PDMParser(pdm_path)
    calc_params = CalcParams(pdm_parser)
    face_aligner = OpenFace22FaceAligner(pdm_path, sim_scale=0.7, output_size=(112, 112))
    triangulation = TriangulationParser("pyfaceau/weights/tris_68_full.txt")
    au_models = OF22ModelParser("pyfaceau/weights/AU_predictors").load_all_models(
        use_recommended=True, use_combined=True, verbose=False
    )

    print(f"  MTCNN backend: {detector.backend_name}")
    print("  Pipeline initialized")

    return {
        'detector': detector,
        'clnf': clnf,
        'pdm_parser': pdm_parser,
        'calc_params': calc_params,
        'face_aligner': face_aligner,
        'triangulation': triangulation,
        'au_models': au_models,
        'fhog': fhog,
    }


AU_NAMES = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09',
            'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23',
            'AU25', 'AU26', 'AU45']


def process_frame(frame, pipeline):
    """Process a single frame through the full pipeline."""
    detector = pipeline['detector']
    clnf = pipeline['clnf']
    calc_params = pipeline['calc_params']
    pdm_parser = pipeline['pdm_parser']
    face_aligner = pipeline['face_aligner']
    triangulation = pipeline['triangulation']
    au_models = pipeline['au_models']
    fhog = pipeline['fhog']

    # Face detection
    bboxes, _ = detector.detect(frame)
    if bboxes is None or len(bboxes) == 0:
        return None
    bbox = bboxes[0][:4].astype(np.float32)

    # CLNF landmarks
    try:
        landmarks, info = clnf.fit(frame, bbox)
        landmarks = landmarks.astype(np.float32)
    except Exception:
        return None

    # Pose estimation
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
    hog = fhog.extract_fhog_features(aligned_face, cell_size=8)
    hog_features = hog.reshape(12, 12, 31).transpose(1, 0, 2).flatten().astype(np.float32)

    # Geometric features
    geom_features = pdm_parser.extract_geometric_features(local_params).astype(np.float32)

    # AU prediction (simplified - without running median for parallel processing)
    full_vector = np.concatenate([hog_features, geom_features])
    au_values = []
    for au_name in AU_NAMES:
        if au_name not in au_models:
            au_values.append(0.0)
            continue
        model = au_models[au_name]
        centered = full_vector - model['means'].flatten()
        pred = np.dot(centered.reshape(1, -1), model['support_vectors']) + model['bias']
        au_values.append(float(np.clip(pred[0, 0], 0.0, 5.0)))
    au_intensities = np.array(au_values, dtype=np.float32)

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


def main():
    parser = argparse.ArgumentParser(description='Process a single video')
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    video_path = Path(args.video_path)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print(f"Processing: {video_path.name}")
    print("=" * 60)

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "landmarks").mkdir(exist_ok=True)
    (output_dir / "hog_features").mkdir(exist_ok=True)

    # Initialize pipeline
    pipeline = init_pipeline()

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    # Process frames
    annotations = []
    saved = 0
    failed = 0
    start_time = time.time()

    for frame_idx in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        result = process_frame(frame, pipeline)

        if result is None:
            failed += 1
            continue

        sample_id = f"{frame_idx:05d}"

        # Save outputs
        cv2.imwrite(str(output_dir / "images" / f"{sample_id}.png"),
                    cv2.cvtColor(result['image'], cv2.COLOR_RGB2BGR))
        np.save(output_dir / "landmarks" / f"{sample_id}.npy", result['landmarks'])
        np.save(output_dir / "hog_features" / f"{sample_id}.npy", result['hog_features'])

        # Build annotation
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

        annotations.append(row)
        saved += 1

    cap.release()
    elapsed = time.time() - start_time

    # Save annotations
    df = pd.DataFrame(annotations)
    df.to_csv(output_dir / "annotations.csv", index=False)

    # Summary
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Video: {video_path.name}")
    print(f"Saved: {saved}/{total_frames} frames")
    print(f"Failed: {failed} frames")
    print(f"Time: {elapsed:.1f}s ({saved/elapsed:.1f} fps)")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
