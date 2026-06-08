# Dataset manifest (provenance) — latest snapshot

Patient data is archived LOCALLY (not in git). This file records checksums
so a code commit maps to a reproducible data state. Roll back by extracting
the tarball named below.

| field | value |
|---|---|
| snapshot | `20260607_210749` |
| note | post smiling-baseline fix: relocated 8 to mid-clip neutral, flagged 10 smiling; 18 BL nodes pending re-review |
| code commit | `5db74db3327b46357b8de394ca75ff992723cb4b` (s25-autocurator-handoff) |
| v1316 CSVs | 222 files |
| v1316 rollup sha256 | `93d5272291bc064370ed3f8bbed036fb4f72165e317e2113aefff6ac1a21b08c` |
| curation sha256 | `7e2f330d70859301b8d4da277337b7fa7303374011ef2b5318685166daf27eb3` |
| params sha256 | `9ba763eb043bbb0b57bc23168a2b2da00bb9ee2586c6748d6da67db6db6efa09` |
| Combined Data mirrored CSVs | 102 files |
| local archive | `/Users/johnwilsoniv/Documents/SplitFace/dataset_snapshots/dataset_20260607_210749.tar.gz` (40M) |
| archive sha256 | `d814e55e8aea428e0c252283ecf56c7a86e912df6fcf60dfb5fdf0ed8fb7ac9a` |

_Regenerate: `bash 'S3 Data Analysis/snapshot_dataset.sh' "note"`_
