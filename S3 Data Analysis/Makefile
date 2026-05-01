# SplitFace S3 — paralysis training pipeline + regression-test framework.
#
# Conventional targets:
#   make help              Show this listing
#   make tier0             Fast determinism gate (≤30s) — run before every commit
#   make tier1             Quality vs C++ gold on 10 canaries (≤10min)
#   make tier2             Release-gate retrain reproducibility (≤15min)
#   make test              tier0 + tier1
#   make test-all          tier0 + tier1 + tier2
#
#   make preflight         Verify goldens + Tier 0 pass before retraining
#   make preflight-retrain Same as preflight, then nudge to next step
#
#   make goldens           Regenerate fast goldens (AUs, peak frames, features,
#                          predictions, test split, metric bands, landmarks
#                          snapshot — does NOT reinstrument pyfaceau).
#   make instrument        Slow: rerun pyfaceau on all 20 hemiface canary videos
#                          (~80 min). Required after pyfaceau code change.
#   make retrain-bands     Slow: re-measure Tier 2 bands (~20 min). Required
#                          after sklearn / xgboost upgrade.
#
#   make install-hooks     Wire up the pre-commit hook (one-time per checkout)
#   make uninstall-hooks   Remove it
#
#   make clean             Remove pytest cache + __pycache__ + scratch logs
#
# Conventions:
#   - PYTHONHASHSEED=42, OMP_NUM_THREADS=1 are passed to every test invocation.
#     These are the determinism env vars the framework expects.
#   - PYTHON variable defaults to the venv at ../.venv/bin/python; override on
#     the command line if you need a different interpreter.
#   - Tier markers (tier0/tier1/tier2) are pytest markers defined in pytest.ini.

PYTHON   ?= ../.venv/bin/python
PYTEST    = $(PYTHON) -m pytest
TEST_ENV  = PYTHONHASHSEED=42 OMP_NUM_THREADS=1
PYTEST_FLAGS ?= -ra --tb=short

# Reset terminal color at the end of every help line
BLUE  := \033[36m
RESET := \033[0m

.PHONY: help tier0 tier1 tier2 test test-all \
        preflight preflight-retrain \
        goldens instrument retrain-bands \
        install-hooks uninstall-hooks \
        clean

help:
	@echo ""
	@echo "SplitFace S3 — make targets:"
	@echo ""
	@printf "  $(BLUE)%-22s$(RESET) %s\n" "tier0"             "Determinism invariants (≤30s)"
	@printf "  $(BLUE)%-22s$(RESET) %s\n" "tier1"             "Quality vs C++ gold (≤10min)"
	@printf "  $(BLUE)%-22s$(RESET) %s\n" "tier2"             "Release-gate retrain reproducibility (≤15min)"
	@printf "  $(BLUE)%-22s$(RESET) %s\n" "test"              "= tier0 + tier1"
	@printf "  $(BLUE)%-22s$(RESET) %s\n" "test-all"          "= tier0 + tier1 + tier2"
	@echo ""
	@printf "  $(BLUE)%-22s$(RESET) %s\n" "preflight"         "Verify goldens + Tier 0 pass before retraining"
	@printf "  $(BLUE)%-22s$(RESET) %s\n" "preflight-retrain" "preflight + show next step"
	@echo ""
	@printf "  $(BLUE)%-22s$(RESET) %s\n" "goldens"           "Regen fast goldens (~10s; not pyfaceau capture)"
	@printf "  $(BLUE)%-22s$(RESET) %s\n" "instrument"        "Re-run pyfaceau on canaries (~80min, slow)"
	@printf "  $(BLUE)%-22s$(RESET) %s\n" "retrain-bands"     "Re-measure Tier 2 bands (~20min, slow)"
	@echo ""
	@printf "  $(BLUE)%-22s$(RESET) %s\n" "install-hooks"     "Install pre-commit hook (one-time)"
	@printf "  $(BLUE)%-22s$(RESET) %s\n" "uninstall-hooks"   "Remove the pre-commit hook"
	@printf "  $(BLUE)%-22s$(RESET) %s\n" "clean"             "Remove pytest/python caches + scratch"
	@echo ""

# --------------------------------------------------------------------------
# Test invocations
# --------------------------------------------------------------------------

tier0:
	@$(TEST_ENV) $(PYTEST) tests/ -m tier0 $(PYTEST_FLAGS)

tier1:
	@$(TEST_ENV) $(PYTEST) tests/ -m tier1 $(PYTEST_FLAGS)

tier2:
	@$(TEST_ENV) $(PYTEST) tests/ -m tier2 $(PYTEST_FLAGS)

test:
	@$(TEST_ENV) $(PYTEST) tests/ -m "tier0 or tier1" $(PYTEST_FLAGS)

test-all:
	@$(TEST_ENV) $(PYTEST) tests/ $(PYTEST_FLAGS)

# --------------------------------------------------------------------------
# Pre-retrain safety
# --------------------------------------------------------------------------

preflight:
	@bin/preflight-retrain

preflight-retrain: preflight
	@echo ""
	@echo "Next: pick the zone you want to retrain, e.g."
	@echo "  $(PYTHON) paralysis_training_pipeline.py mid"

# --------------------------------------------------------------------------
# Golden regeneration
# --------------------------------------------------------------------------

goldens:
	@$(TEST_ENV) $(PYTHON) tests/update_goldens.py --stage all \
	    --reason "manual `date +%Y-%m-%d` regen via make goldens"

instrument:
	@echo "Slow: pyfaceau capture on 20 canary hemiface videos (~80 min)..."
	@$(TEST_ENV) $(PYTHON) tests/instrument_pyfaceau.py --all

retrain-bands:
	@echo "Slow: 6 zone × source retrains for Tier 2 baseline (~20 min)..."
	@$(TEST_ENV) $(PYTHON) tests/update_goldens.py --stage retrain_bands \
	    --reason "manual `date +%Y-%m-%d` retrain bands refresh"
	@$(TEST_ENV) $(PYTHON) tests/update_goldens.py --stage metric_bands \
	    --reason "incorporate fresh retrain_bands"

# --------------------------------------------------------------------------
# Git hook installation
# --------------------------------------------------------------------------

# Install hook by copying our template into .git/hooks/.
# Uses repo-rooted .git regardless of cwd.
install-hooks:
	@bin/install-hooks

uninstall-hooks:
	@HOOK_FILE=$$(git rev-parse --show-toplevel)/.git/hooks/pre-commit; \
	 if [ -f $$HOOK_FILE ] && grep -q "splitface-s3-tier0" $$HOOK_FILE; then \
	     rm $$HOOK_FILE; \
	     echo "Removed $$HOOK_FILE"; \
	 else \
	     echo "No splitface-s3 hook to remove (or hook is from a different source)."; \
	 fi

# --------------------------------------------------------------------------
# Housekeeping
# --------------------------------------------------------------------------

clean:
	@find tests -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@rm -rf tests/.pytest_cache
	@rm -rf tests/_tier2_artifacts tests/_retrain_bands_logs
	@rm -f tests/_tier2_config_override.py tests/_retrain_bands_helper.py
	@echo "Cleaned."
