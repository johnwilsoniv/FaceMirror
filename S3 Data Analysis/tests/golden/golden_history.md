# Golden Update History

Append-only log of every `update_goldens.py` run.

## 2026-04-29T15:16:10+00:00
- **Stage(s):** all
- **Reason:** initial baseline (Sub-PR 1)
- **Git SHA:** `fecf2f9bc98dc74ec91fc9214c958fa02fed76af`
- **pip-freeze SHA256:** `e0661c83e53e6f40563234155dfbfdc1634deb31877ec8165fe901e3f97d08d5`
- **Files written:** 67

## 2026-04-29T15:18:18+00:00
- **Stage(s):** aus
- **Reason:** fix: store full AU snapshots (not inner-joined) so Tier 0 byte-equality works
- **Git SHA:** `fecf2f9bc98dc74ec91fc9214c958fa02fed76af`
- **pip-freeze SHA256:** `e0661c83e53e6f40563234155dfbfdc1634deb31877ec8165fe901e3f97d08d5`
- **Files written:** 40

## 2026-04-29T15:21:33+00:00
- **Stage(s):** metric_bands
- **Reason:** auto-calibrate bands from current observations
- **Git SHA:** `fecf2f9bc98dc74ec91fc9214c958fa02fed76af`
- **pip-freeze SHA256:** `e0661c83e53e6f40563234155dfbfdc1634deb31877ec8165fe901e3f97d08d5`
- **Files written:** 1

## 2026-04-29T15:23:22+00:00
- **Stage(s):** all
- **Reason:** idempotency check
- **Git SHA:** `fecf2f9bc98dc74ec91fc9214c958fa02fed76af`
- **pip-freeze SHA256:** `e0661c83e53e6f40563234155dfbfdc1634deb31877ec8165fe901e3f97d08d5`
- **Files written:** 67

## 2026-04-29T15:39:35+00:00
- **Stage(s):** landmarks
- **Reason:** snapshot landmarks (only 1/20 pyfaceau parquets so far; rest queued)
- **Git SHA:** `5ded846746cc79a49add034944c067dbf205a74f`
- **pip-freeze SHA256:** `e0661c83e53e6f40563234155dfbfdc1634deb31877ec8165fe901e3f97d08d5`
- **Files written:** 20

## 2026-04-29T15:39:47+00:00
- **Stage(s):** metric_bands
- **Reason:** calibrate from 1 pyfaceau parquet
- **Git SHA:** `5ded846746cc79a49add034944c067dbf205a74f`
- **pip-freeze SHA256:** `e0661c83e53e6f40563234155dfbfdc1634deb31877ec8165fe901e3f97d08d5`
- **Files written:** 1

## 2026-04-29T16:17:23+00:00
- **Stage(s):** retrain_bands
- **Reason:** complete Tier 2 baseline (samefile fix)
- **Git SHA:** `5ded846746cc79a49add034944c067dbf205a74f`
- **pip-freeze SHA256:** `e0661c83e53e6f40563234155dfbfdc1634deb31877ec8165fe901e3f97d08d5`
- **Files written:** 1

## 2026-04-29T16:17:39+00:00
- **Stage(s):** metric_bands
- **Reason:** incorporate retrain_bands measurements
- **Git SHA:** `5ded846746cc79a49add034944c067dbf205a74f`
- **pip-freeze SHA256:** `e0661c83e53e6f40563234155dfbfdc1634deb31877ec8165fe901e3f97d08d5`
- **Files written:** 1

## 2026-04-29T16:18:19+00:00
- **Stage(s):** metric_bands
- **Reason:** widen Tier 2 bands to ±0.06 for stochasticity
- **Git SHA:** `5ded846746cc79a49add034944c067dbf205a74f`
- **pip-freeze SHA256:** `e0661c83e53e6f40563234155dfbfdc1634deb31877ec8165fe901e3f97d08d5`
- **Files written:** 1

## 2026-04-29T17:03:26+00:00
- **Stage(s):** production_predictions
- **Reason:** initial Stage 7 baseline
- **Git SHA:** `51887b55a327b19f5fd150a5a8016bd65c0325f3`
- **pip-freeze SHA256:** `e0661c83e53e6f40563234155dfbfdc1634deb31877ec8165fe901e3f97d08d5`
- **Files written:** 1

## 2026-04-29T17:19:00+00:00
- **Stage(s):** retrain_bands
- **Reason:** PYTHONSTARTUP bug fixed; re-measure deterministic bands
- **Git SHA:** `71fe1c9c6b16ff1bf4703905c27d6d6e680bbf5d`
- **pip-freeze SHA256:** `e0661c83e53e6f40563234155dfbfdc1634deb31877ec8165fe901e3f97d08d5`
- **Files written:** 1

## 2026-04-29T17:19:45+00:00
- **Stage(s):** metric_bands
- **Reason:** tight Tier 2 bands now that wrapper bug is fixed
- **Git SHA:** `71fe1c9c6b16ff1bf4703905c27d6d6e680bbf5d`
- **pip-freeze SHA256:** `e0661c83e53e6f40563234155dfbfdc1634deb31877ec8165fe901e3f97d08d5`
- **Files written:** 1

## 2026-04-29T17:44:33+00:00
- **Stage(s):** all
- **Reason:** all 20 pyfaceau parquets captured; final calibration
- **Git SHA:** `c3937ad8d8fc2abda0e9620bcfe371e56c1c503d`
- **pip-freeze SHA256:** `e0661c83e53e6f40563234155dfbfdc1634deb31877ec8165fe901e3f97d08d5`
- **Files written:** 68

## 2026-04-29T18:19:49+00:00
- **Stage(s):** gpu_divergence
- **Reason:** initial GPU vs CPU divergence baseline on IMG_0942 left
- **Git SHA:** `fa60e4cfc5f4f7c6efcb6521d0150891d529d8d6`
- **pip-freeze SHA256:** `e0661c83e53e6f40563234155dfbfdc1634deb31877ec8165fe901e3f97d08d5`
- **Files written:** 1

## 2026-04-29T19:45:23+00:00
- **Stage(s):** batch_processor_subset
- **Reason:** initial baseline lock
- **Git SHA:** `fa60e4cfc5f4f7c6efcb6521d0150891d529d8d6`
- **pip-freeze SHA256:** `e0661c83e53e6f40563234155dfbfdc1634deb31877ec8165fe901e3f97d08d5`
- **Files written:** 1

## 2026-05-01T17:10:59+00:00
- **Stage(s):** all
- **Reason:** production-2026-05-01 milestone snapshot
- **Git SHA:** `b2a8aab38d74bd5dab0c9cd3cdbcebafcb55469e`
- **pip-freeze SHA256:** `54d6d5b2d372c1f2d3c99d1bf8ca91dae891cfde25d5486a6ee4bea55b49b2ea`
- **Files written:** 68

## 2026-05-02T17:43:25+00:00
- **Stage(s):** windows_cuda_aus
- **Reason:** patched pyfhog v0.1.4 (HOG indexing fix)
- **Git SHA:** `a4998fe509852221ef34520dda6738863300e516`
- **pip-freeze SHA256:** `68e13498d25b31da1a2dff225fcfedcc37ad6b319e771688b4fa819903576c02`
- **Files written:** 20

## 2026-05-02T20:03:13+00:00
- **Stage(s):** metric_bands
- **Reason:** include Windows-CUDA observations alongside macOS in stage3 band calibration
- **Git SHA:** `1558c80852100d184cf71c539a17db6df195b767`
- **pip-freeze SHA256:** `68e13498d25b31da1a2dff225fcfedcc37ad6b319e771688b4fa819903576c02`
- **Files written:** 1

