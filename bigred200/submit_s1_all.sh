#!/bin/bash
# Submit S1 Face Mirror job for all 111 videos
#
# Usage:
#   ./submit_s1_all.sh           # Submit all 111 videos
#   ./submit_s1_all.sh 0-10      # Submit only videos 0-10
#   ./submit_s1_all.sh 50        # Submit only video 50

cd $HOME/SplitFace

# Create output and log directories
mkdir -p output/s1_processed
mkdir -p logs

if [ -n "$1" ]; then
    # Custom array range
    echo "Submitting S1 job for videos: $1"
    sbatch --array=$1 bigred200/s1_all_videos.slurm
else
    # All videos
    echo "Submitting S1 job for all 111 videos..."
    sbatch bigred200/s1_all_videos.slurm
fi

echo ""
echo "Monitor with: squeue -u $USER"
echo "Check logs:   tail -f logs/s1_*.out"
