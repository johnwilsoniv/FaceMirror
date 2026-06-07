# Dataset manifest (provenance) — latest snapshot

Patient data is archived LOCALLY (not in git). This file records checksums
so a code commit maps to a reproducible data state. Roll back by extracting
the tarball named below.

| field | value |
|---|---|
| snapshot | `20260607_111644` |
| note | initial snapshot: 30-patient cohort, auto-curator v2 deployed |
| code commit | `d4802f1f8cbe06501a0e745dfe3a94cc05dfbeb2` (s25-autocurator-handoff) |
| v1316 CSVs | 222 files |
| v1316 rollup sha256 | `ba0a6d6ca4df8d5b8cdc9224da97be0780cc18c64f8789fbc7daebc3f0d79774` |
| curation sha256 | `86e72bdd0996ee49fe9464729178c0d78445a612350ebc78e8cf0b350aea758c` |
| params sha256 | `7f756468c8581eb86d855a2a196d14a3bb15cae97d3e801b8ac04f9eedc0bd1b` |
| Combined Data mirrored CSVs | 96 files |
| local archive | `/Users/johnwilsoniv/Documents/SplitFace/dataset_snapshots/dataset_20260607_111644.tar.gz` (35M) |
| archive sha256 | `080459a6479dea58c53e6a73c5f573594848998ebcf2bc4de092c65411bd82b0` |

_Regenerate: `bash 'S3 Data Analysis/snapshot_dataset.sh' "note"`_
