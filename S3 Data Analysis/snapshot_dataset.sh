#!/bin/bash
# snapshot_dataset.sh — versioned snapshot of the S2.5 pilot dataset.
#
# PRIVACY: the dataset is PATIENT DATA and is gitignored (pre-publication). This
# script does NOT push it to any remote. It writes a LOCAL timestamped archive and
# a checksummed manifest. Only the MANIFEST (hashes + metadata, no patient data) is
# committed to git, tying each code commit to a reproducible data state. Roll back
# by extracting the matching tarball.
#
# Usage:  bash "S3 Data Analysis/snapshot_dataset.sh" ["note"]
set -euo pipefail

REPO="/Users/johnwilsoniv/Documents/SplitFace Open3"
SNAPDIR="/Users/johnwilsoniv/Documents/SplitFace/dataset_snapshots"          # LOCAL, not in repo
MANIFEST="$REPO/S3 Data Analysis/DATASET_MANIFEST.md"                         # committed (hashes only)
V1316="$REPO/S3 Data Analysis/recoded_rerun_dual_v1316"
CURATION="/Users/johnwilsoniv/Documents/SplitFace/S25 Curated Files/s25_curation.json"
COMBINED="/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data"
PARAMS="$REPO/S3 Data Analysis/s25_auto_params.json"
NOTE="${1:-}"

TS=$(date +%Y%m%d_%H%M%S)
GIT_SHA=$(cd "$REPO" && git rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_BRANCH=$(cd "$REPO" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
mkdir -p "$SNAPDIR"

# null-delimited throughout so paths with spaces ("SplitFace Open3") are safe
rollup() { find "$1" -name '*.csv' -print0 2>/dev/null | LC_ALL=C sort -z | xargs -0 shasum -a 256 2>/dev/null | shasum -a 256 | awk '{print $1}'; }
hashof() { [ -f "$1" ] && shasum -a 256 "$1" | awk '{print $1}' || echo "MISSING"; }

V1316_N=$(find "$V1316" -name '*_coded.csv' 2>/dev/null | wc -l | xargs)
V1316_ROLLUP=$(rollup "$V1316")
CURATION_H=$(hashof "$CURATION")
PARAMS_H=$(hashof "$PARAMS")
MIRRORED_N=$(find "$COMBINED" -name '*_mirrored.csv' 2>/dev/null | wc -l | xargs)

# --- LOCAL archive (patient data) ---
TARBALL="$SNAPDIR/dataset_$TS.tar.gz"
tar -czf "$TARBALL" \
  -C "$REPO/S3 Data Analysis" recoded_rerun_dual_v1316 \
  -C "/Users/johnwilsoniv/Documents/SplitFace/S25 Curated Files" s25_curation.json \
  2>/dev/null || true
TAR_H=$(hashof "$TARBALL")
TAR_SZ=$(du -h "$TARBALL" 2>/dev/null | awk '{print $1}')

# --- committed manifest (hashes only) ---
{
  echo "# Dataset manifest (provenance) — latest snapshot"
  echo
  echo "Patient data is archived LOCALLY (not in git). This file records checksums"
  echo "so a code commit maps to a reproducible data state. Roll back by extracting"
  echo "the tarball named below."
  echo
  echo "| field | value |"
  echo "|---|---|"
  echo "| snapshot | \`$TS\` |"
  echo "| note | ${NOTE:-—} |"
  echo "| code commit | \`$GIT_SHA\` ($GIT_BRANCH) |"
  echo "| v1316 CSVs | $V1316_N files |"
  echo "| v1316 rollup sha256 | \`$V1316_ROLLUP\` |"
  echo "| curation sha256 | \`$CURATION_H\` |"
  echo "| params sha256 | \`$PARAMS_H\` |"
  echo "| Combined Data mirrored CSVs | $MIRRORED_N files |"
  echo "| local archive | \`$TARBALL\` ($TAR_SZ) |"
  echo "| archive sha256 | \`$TAR_H\` |"
  echo
  echo "_Regenerate: \`bash 'S3 Data Analysis/snapshot_dataset.sh' \"note\"\`_"
} > "$MANIFEST"

echo "Snapshot $TS written:"
echo "  archive  -> $TARBALL ($TAR_SZ)"
echo "  manifest -> $MANIFEST (commit this)"
echo "  v1316 rollup: $V1316_ROLLUP"
