# PyFaceAU HPC Pipeline for Big Red 200

High-performance AU extraction pipeline optimized for Indiana University's Big Red 200 supercomputer.

## Optimization Phases

This pipeline incorporates five optimization phases to achieve maximum throughput on AMD EPYC processors:

| Phase | Optimization | Impact |
|-------|-------------|--------|
| 1 | Shared Memory Model Loading | 93% memory reduction (424MB × N → 424MB × 1) |
| 2 | CLNF Convergence Profiles | 30-50% faster landmark detection |
| 3 | NUMA-Aware Worker Pool | 10-30% reduced memory latency |
| 4 | Float32 Data Types | 50% memory bandwidth savings |
| 5 | Zero-Copy Frame Buffers | Eliminates serialization overhead |

## Quick Start

### 1. Transfer code to Big Red 200:
```bash
rsync -avz --exclude='Patient Data' --exclude='.git' --exclude='__pycache__' \
    ~/Documents/SplitFace\ Open3/ username@bigred200.uits.iu.edu:~/pyfaceau/
```

### 2. Transfer video data:
```bash
rsync -avz ~/Documents/SplitFace\ Open3/Patient\ Data/ \
    username@bigred200.uits.iu.edu:~/pyfaceau/Patient\ Data/
```

### 3. SSH and submit job:
```bash
ssh username@bigred200.uits.iu.edu
cd ~/pyfaceau/bigred200

# Single video
sbatch submit_hpc_job.sh "S Data/Normal Cohort/IMG_0942.MOV"

# Entire directory
sbatch submit_hpc_job.sh "S Data/Normal Cohort/" ./results/
```

## File Structure

```
bigred200/
├── hpc_au_pipeline.py      # Main HPC-optimized pipeline
├── numa_worker_pool.py     # NUMA-aware multiprocessing
├── shared_memory_init.py   # Shared memory initialization
├── submit_hpc_job.sh       # SLURM submission script
├── run_validation_hpc.py   # Validation against C++ OpenFace
├── setup_environment.sh    # One-time environment setup
└── README.md               # This file
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `N_WORKERS` | 64 | Number of parallel workers |
| `CONVERGENCE_PROFILE` | optimized | CLNF convergence profile |
| `USE_NUMA` | 1 | Enable NUMA-aware placement |
| `USE_SHARED_MEMORY` | 1 | Enable shared memory models |

### Convergence Profiles

| Profile | Speed | Accuracy | Use Case |
|---------|-------|----------|----------|
| `accurate` | 1.0x | Reference | Validation, benchmarking |
| `optimized` | 1.4x | <0.3px loss | **Recommended for production** |
| `fast` | 2.0x | <1.0px loss | Real-time preview |
| `video` | 1.8x | <0.5px loss | Sequential video processing |

## Performance Expectations

### Big Red 200 (128 cores, 256GB RAM)

| Configuration | Expected FPS | Memory Usage |
|--------------|--------------|--------------|
| Baseline (16 workers) | ~16 FPS | ~7 GB |
| Optimized (64 workers) | ~50-80 FPS | ~2 GB |
| Maximum (128 workers) | ~80-120 FPS | ~3 GB |

## First Time Setup

Run once to create the conda environment:
```bash
cd ~/pyfaceau/bigred200
bash setup_environment.sh
```

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View job output (replace JOBID)
tail -f logs/pyfaceau_JOBID.out

# Cancel a job
scancel JOBID
```

## Python API

```python
from bigred200.hpc_au_pipeline import HPCAUPipeline, HPCConfig

config = HPCConfig(
    n_workers=64,
    convergence_profile='optimized',
    use_shared_memory=True,
    use_numa=True
)

pipeline = HPCAUPipeline(config)
df = pipeline.process_video('video.mp4', 'output.csv')
```

## Troubleshooting

### Out of Memory
Reduce workers: `N_WORKERS=32 sbatch submit_hpc_job.sh video.mp4`

### Shared Memory Errors
```bash
df -h /dev/shm  # Check space
rm -rf /dev/shm/pyclnf_models /dev/shm/pyfaceau_models  # Cleanup
```

## Resources

- [Big Red 200 Documentation](https://kb.iu.edu/d/brcc)
- [SLURM Job Submission](https://kb.iu.edu/d/awrz)
- [GPU Jobs on Big Red 200](https://kb.iu.edu/d/avjk)
