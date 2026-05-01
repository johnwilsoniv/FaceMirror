#!/usr/bin/env bash
# splitface-s3-tier0 pre-commit hook — installed by S3 Data Analysis/bin/install-hooks.
# Runs Tier 0 (deterministic invariants, ~3s) when this commit touches files
# under "S3 Data Analysis/". For other changes, exits silently.
#
# Bypass for one-off emergency: git commit --no-verify
# Remove: make uninstall-hooks

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
S3_DIR="$REPO_ROOT/S3 Data Analysis"

# Files staged for this commit, restricted to S3 Data Analysis
S3_CHANGED=$(git diff --cached --name-only --diff-filter=ACMR | grep -E "^S3 Data Analysis/" || true)
if [[ -z "$S3_CHANGED" ]]; then
    exit 0
fi

# Skip the hook if pyfaceau/pyclnf packages aren't importable (e.g. checking
# out an older branch with different deps). Better to silently skip than to
# falsely fail the commit.
if ! "$S3_DIR/../.venv/bin/python" -c "import pyfaceau, pyclnf" 2>/dev/null; then
    echo "[pre-commit] pyfaceau/pyclnf not available in venv — skipping S3 Tier 0."
    exit 0
fi

printf "[pre-commit] Running S3 Tier 0 (≤3s)...\n"
if ! ( cd "$S3_DIR" && make tier0 PYTEST_FLAGS="-q --tb=line --no-header" 2>&1 | tail -20 ); then
    printf "\n[pre-commit] Tier 0 FAILED. Fix the regression or commit with --no-verify (and explain why in the message).\n"
    exit 1
fi
exit 0
