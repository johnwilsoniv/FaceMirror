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

