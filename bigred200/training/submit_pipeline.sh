#!/bin/bash
# ============================================
# Master Script: Training Data Generation & Neural Network Training Pipeline
# ============================================
#
# This script orchestrates the complete training pipeline on Big Red 200:
# 1. Generate video manifest
# 2. Submit array job for per-video HDF5 generation
# 3. Submit merge job (depends on array completion)
# 4. Optionally submit GPU training jobs
#
# Usage:
#   bash submit_pipeline.sh                    # Full pipeline
#   bash submit_pipeline.sh --data-only        # Only data generation
#   bash submit_pipeline.sh --train-only       # Only training (assumes data exists)
#
# ============================================

set -e

# Parse arguments
DATA_ONLY=false
TRAIN_ONLY=false

for arg in "$@"; do
    case $arg in
        --data-only)
            DATA_ONLY=true
            shift
            ;;
        --train-only)
            TRAIN_ONLY=true
            shift
            ;;
        --help)
            echo "Usage: bash submit_pipeline.sh [--data-only|--train-only]"
            echo ""
            echo "Options:"
            echo "  --data-only   Only generate training data (skip NN training)"
            echo "  --train-only  Only submit training jobs (assumes data exists)"
            exit 0
            ;;
    esac
done

echo "=============================================="
echo "Training Pipeline Submission"
echo "=============================================="
echo ""

cd $HOME/pyfaceau

# ============================================
# STEP 1: Generate Video Manifest
# ============================================

if [ "$TRAIN_ONLY" = false ]; then
    echo "Step 1: Generating video manifest..."
    python bigred200/training/generate_video_manifest.py \
        --videos-dir "S Data" \
        --output bigred200/config/video_manifest.txt

    VIDEO_COUNT=$(wc -l < bigred200/config/video_manifest.txt)
    echo "Found $VIDEO_COUNT videos"
    echo ""

    # Create output directories
    echo "Step 2: Creating output directories..."
    mkdir -p bigred200/output/{per_video,logs,checkpoints}
    echo ""

    # ============================================
    # STEP 2: Submit Array Job for Data Generation
    # ============================================

    echo "Step 3: Submitting data generation array job..."
    ARRAY_MAX=$((VIDEO_COUNT - 1))

    # Update array size in SLURM script dynamically
    ARRAY_JOB_ID=$(sbatch --parsable \
        --array=0-${ARRAY_MAX}%80 \
        bigred200/training/generate_training_array.slurm)

    echo "Array job submitted: $ARRAY_JOB_ID"
    echo "  Range: 0-$ARRAY_MAX (max 80 concurrent)"
    echo ""

    # ============================================
    # STEP 3: Submit Merge Job (with dependency)
    # ============================================

    echo "Step 4: Submitting merge job..."
    MERGE_JOB_ID=$(sbatch --parsable \
        --dependency=afterany:$ARRAY_JOB_ID \
        bigred200/training/merge_training_data.slurm)

    echo "Merge job submitted: $MERGE_JOB_ID"
    echo "  Depends on: $ARRAY_JOB_ID"
    echo ""

    if [ "$DATA_ONLY" = true ]; then
        echo "=============================================="
        echo "Data Generation Pipeline Submitted (--data-only)"
        echo "=============================================="
        echo ""
        echo "Jobs submitted:"
        echo "  Array job: $ARRAY_JOB_ID (0-$ARRAY_MAX)"
        echo "  Merge job: $MERGE_JOB_ID"
        echo ""
        echo "Monitor with:"
        echo "  squeue -u \$USER"
        echo ""
        echo "After completion, run training with:"
        echo "  bash bigred200/training/submit_pipeline.sh --train-only"
        exit 0
    fi

    TRAIN_DEPENDENCY="--dependency=afterok:$MERGE_JOB_ID"
else
    # Train-only mode
    if [ ! -f "bigred200/output/training_data_final.h5" ]; then
        echo "ERROR: Training data not found!"
        echo "  Expected: bigred200/output/training_data_final.h5"
        echo ""
        echo "Run data generation first:"
        echo "  bash bigred200/training/submit_pipeline.sh --data-only"
        exit 1
    fi

    echo "Step 1: Skipping data generation (--train-only)"
    echo "  Using: bigred200/output/training_data_final.h5"
    echo ""
    TRAIN_DEPENDENCY=""
fi

# ============================================
# STEP 4: Submit GPU Training Jobs
# ============================================

echo "Step 5: Submitting GPU training jobs..."

# AU MLP Training
AU_JOB_ID=$(sbatch --parsable \
    $TRAIN_DEPENDENCY \
    bigred200/training/train_au_mlp.slurm)

echo "AU MLP job submitted: $AU_JOB_ID"

# Landmark/Pose Training
LP_JOB_ID=$(sbatch --parsable \
    $TRAIN_DEPENDENCY \
    bigred200/training/train_landmark_pose.slurm)

echo "Landmark/Pose job submitted: $LP_JOB_ID"
echo ""

# ============================================
# Summary
# ============================================

echo "=============================================="
echo "Pipeline Submitted Successfully!"
echo "=============================================="
echo ""

if [ "$TRAIN_ONLY" = false ]; then
    echo "Jobs submitted:"
    echo "  1. Data Generation: $ARRAY_JOB_ID (array job)"
    echo "  2. Data Merge:      $MERGE_JOB_ID"
    echo "  3. AU MLP Training: $AU_JOB_ID (GPU)"
    echo "  4. Landmark/Pose:   $LP_JOB_ID (GPU)"
else
    echo "Jobs submitted:"
    echo "  1. AU MLP Training: $AU_JOB_ID (GPU)"
    echo "  2. Landmark/Pose:   $LP_JOB_ID (GPU)"
fi

echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f bigred200/output/logs/*.out"
echo ""
echo "TensorBoard (after training starts):"
echo "  ssh -L 6006:localhost:6006 \$USER@bigred200.uits.iu.edu"
echo "  Then open http://localhost:6006"
echo ""
echo "Expected outputs:"
echo "  - bigred200/output/training_data_final.h5"
echo "  - models/au_mlp/au_mlp_best.pt"
echo "  - models/landmark_pose/checkpoint_best.pt"
echo "  - models/landmark_pose/landmark_pose.onnx"
