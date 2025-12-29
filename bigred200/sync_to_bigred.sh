#!/bin/bash
# Sync SplitFace project to BigRed200
#
# Usage: ./sync_to_bigred.sh

REMOTE="bigred200:~/SplitFace"

echo "Syncing to BigRed200..."
echo ""

# Sync main directories (exclude large/temporary files)
rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude '.DS_Store' \
    --exclude 'output/' \
    --exclude 'logs/' \
    --exclude '*.mp4' \
    --exclude '*.MOV' \
    --exclude '*.mov' \
    --exclude 'archive/' \
    --exclude 'validation_output/' \
    --exclude 'benchmark_output/' \
    "/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf/" \
    "$REMOTE/pyclnf/"

rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude '.DS_Store' \
    "/Users/johnwilsoniv/Documents/SplitFace Open3/pymtcnn/" \
    "$REMOTE/pymtcnn/"

rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude '.DS_Store' \
    --exclude 'models/' \
    --exclude 'nnclnf_training_data/' \
    "/Users/johnwilsoniv/Documents/SplitFace Open3/pyfaceau/" \
    "$REMOTE/pyfaceau/"

rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude '.DS_Store' \
    "/Users/johnwilsoniv/Documents/SplitFace Open3/pyfhog/" \
    "$REMOTE/pyfhog/"

rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude '.DS_Store' \
    "/Users/johnwilsoniv/Documents/SplitFace Open3/S1_FaceMirror/" \
    "$REMOTE/S1_FaceMirror/"

rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude '.DS_Store' \
    "/Users/johnwilsoniv/Documents/SplitFace Open3/bigred200/" \
    "$REMOTE/bigred200/"

echo ""
echo "Code synced. Now sync videos with:"
echo "  rsync -avz --progress '/Users/johnwilsoniv/Documents/SplitFace Open3/S Data/' bigred200:~/SplitFace/'S Data/'"
echo ""
echo "Then on BigRed200:"
echo "  cd ~/SplitFace"
echo "  chmod +x bigred200/submit_s1_all.sh"
echo "  ./bigred200/submit_s1_all.sh"
