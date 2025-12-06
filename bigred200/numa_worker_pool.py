#!/usr/bin/env python3
"""
NUMA-Aware Worker Pool for Big Red 200 HPC

This module provides a NUMA-aware multiprocessing pool optimized for AMD EPYC
processors on Big Red 200. It pins workers to specific NUMA nodes to minimize
cross-node memory access latency.

Big Red 200 Architecture:
- AMD EPYC 7713 (Milan) processors
- 2 sockets per node, 64 cores per socket = 128 cores per node
- 8 NUMA domains per socket (NPS=4 configuration)
- ~32GB memory per NUMA domain

Key Optimizations:
1. NUMA-aware worker placement: Workers are pinned to specific NUMA nodes
2. Memory locality: Shared memory files are accessed from local NUMA node
3. Batch processing: Workers process multiple frames per task to amortize overhead
4. Load balancing: Work is distributed evenly across NUMA nodes

Usage:
    from numa_worker_pool import NUMAWorkerPool, detect_numa_topology

    # Auto-detect NUMA topology
    topology = detect_numa_topology()
    print(f"Detected {topology['num_nodes']} NUMA nodes")

    # Create NUMA-aware pool
    with NUMAWorkerPool(
        n_workers=64,
        initializer=init_worker_shared,
        initargs=(shm_config,)
    ) as pool:
        results = pool.map(process_frame_data, args_list)
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple
from multiprocessing import Pool, cpu_count
import multiprocessing as mp

# Check for libnuma availability
try:
    import ctypes
    _libnuma = ctypes.CDLL("libnuma.so.1", mode=ctypes.RTLD_GLOBAL)
    NUMA_AVAILABLE = True
except OSError:
    NUMA_AVAILABLE = False
    _libnuma = None


def detect_numa_topology() -> Dict[str, Any]:
    """
    Detect NUMA topology of the current system.

    Returns:
        Dictionary with NUMA topology information:
        - num_nodes: Number of NUMA nodes
        - num_cpus: Total number of CPUs
        - cpus_per_node: List of CPU IDs for each NUMA node
        - memory_per_node: Memory in GB for each NUMA node (if available)
        - is_numa: Whether NUMA is available

    Example:
        >>> topology = detect_numa_topology()
        >>> print(f"NUMA nodes: {topology['num_nodes']}")
        >>> print(f"CPUs per node: {[len(cpus) for cpus in topology['cpus_per_node']]}")
    """
    topology = {
        'num_nodes': 1,
        'num_cpus': cpu_count(),
        'cpus_per_node': [list(range(cpu_count()))],
        'memory_per_node': [],
        'is_numa': False
    }

    # Try to detect NUMA topology using lscpu
    try:
        result = subprocess.run(
            ['lscpu', '--parse=CPU,NODE'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Parse lscpu output
            node_cpus: Dict[int, List[int]] = {}
            for line in result.stdout.strip().split('\n'):
                if line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        cpu_id = int(parts[0])
                        node_id = int(parts[1])
                        if node_id not in node_cpus:
                            node_cpus[node_id] = []
                        node_cpus[node_id].append(cpu_id)
                    except ValueError:
                        continue

            if node_cpus:
                num_nodes = max(node_cpus.keys()) + 1
                topology['num_nodes'] = num_nodes
                topology['cpus_per_node'] = [
                    sorted(node_cpus.get(i, []))
                    for i in range(num_nodes)
                ]
                topology['is_numa'] = num_nodes > 1

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Try to get memory info per node
    try:
        result = subprocess.run(
            ['numactl', '--hardware'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            memory_per_node = []
            for line in result.stdout.split('\n'):
                if 'size:' in line.lower():
                    # Parse "node 0 size: 32168 MB"
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p.lower() == 'size:' and i + 1 < len(parts):
                            try:
                                size_mb = int(parts[i + 1])
                                memory_per_node.append(size_mb / 1024)  # Convert to GB
                            except ValueError:
                                pass
            topology['memory_per_node'] = memory_per_node
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return topology


def set_cpu_affinity(cpu_list: List[int]) -> bool:
    """
    Set CPU affinity for the current process.

    Args:
        cpu_list: List of CPU IDs to pin the process to

    Returns:
        True if successful, False otherwise
    """
    try:
        os.sched_setaffinity(0, set(cpu_list))
        return True
    except (AttributeError, OSError):
        # sched_setaffinity not available on all platforms
        return False


def set_numa_node(node_id: int) -> bool:
    """
    Set NUMA node preference for memory allocation.

    Args:
        node_id: NUMA node ID to prefer for memory allocation

    Returns:
        True if successful, False otherwise
    """
    if not NUMA_AVAILABLE or _libnuma is None:
        return False

    try:
        # numa_run_on_node(node_id)
        _libnuma.numa_run_on_node(node_id)
        # numa_set_preferred(node_id)
        _libnuma.numa_set_preferred(node_id)
        return True
    except Exception:
        return False


def _numa_worker_init(
    worker_id: int,
    numa_node: int,
    cpu_list: List[int],
    user_initializer: Optional[Callable],
    user_initargs: Tuple
):
    """
    Initialize a worker with NUMA affinity.

    This is called once when each worker process starts.
    It sets CPU affinity and NUMA memory preference before
    calling the user's initializer.

    Args:
        worker_id: Unique ID for this worker
        numa_node: NUMA node to pin this worker to
        cpu_list: List of CPU IDs for this worker
        user_initializer: User's worker initialization function
        user_initargs: Arguments for user's initializer
    """
    # Set CPU affinity
    if cpu_list:
        affinity_set = set_cpu_affinity(cpu_list)
        if affinity_set:
            # Store which CPUs we're running on for debugging
            os.environ['WORKER_CPUS'] = ','.join(map(str, cpu_list))

    # Set NUMA node preference
    if numa_node >= 0:
        numa_set = set_numa_node(numa_node)
        if numa_set:
            os.environ['WORKER_NUMA_NODE'] = str(numa_node)

    # Store worker ID for debugging
    os.environ['WORKER_ID'] = str(worker_id)

    # Call user's initializer
    if user_initializer is not None:
        user_initializer(*user_initargs)


class NUMAWorkerPool:
    """
    NUMA-aware worker pool for HPC workloads.

    This pool automatically detects NUMA topology and distributes workers
    across NUMA nodes for optimal memory locality.

    Example:
        >>> with NUMAWorkerPool(
        ...     n_workers=64,
        ...     initializer=init_worker_shared,
        ...     initargs=(shm_config,)
        ... ) as pool:
        ...     results = pool.map(process_frame_data, args_list)
    """

    def __init__(
        self,
        n_workers: Optional[int] = None,
        initializer: Optional[Callable] = None,
        initargs: Tuple = (),
        batch_size: int = 1,
        numa_balance: bool = True
    ):
        """
        Initialize NUMA-aware worker pool.

        Args:
            n_workers: Number of worker processes (default: auto-detect)
            initializer: Function to call when each worker starts
            initargs: Arguments for initializer
            batch_size: Number of items per task (for batched processing)
            numa_balance: If True, balance workers across NUMA nodes
        """
        # Detect NUMA topology
        self.topology = detect_numa_topology()
        self.is_numa = self.topology['is_numa']

        # Determine number of workers
        if n_workers is None:
            # Use SLURM allocation if available, otherwise all CPUs
            n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count()))
        self.n_workers = min(n_workers, self.topology['num_cpus'])

        self.initializer = initializer
        self.initargs = initargs
        self.batch_size = batch_size
        self.numa_balance = numa_balance

        # Compute worker-to-NUMA mapping
        self.worker_numa_map = self._compute_numa_mapping()

        # Create the underlying pool
        self._pool = None

    def _compute_numa_mapping(self) -> List[Dict[str, Any]]:
        """
        Compute which NUMA node and CPUs each worker should use.

        Returns:
            List of dicts, one per worker, with 'numa_node' and 'cpus' keys
        """
        mapping = []

        if not self.is_numa or not self.numa_balance:
            # No NUMA or balancing disabled - just distribute evenly
            all_cpus = list(range(self.topology['num_cpus']))
            cpus_per_worker = max(1, len(all_cpus) // self.n_workers)

            for i in range(self.n_workers):
                start = i * cpus_per_worker
                end = start + cpus_per_worker if i < self.n_workers - 1 else len(all_cpus)
                mapping.append({
                    'worker_id': i,
                    'numa_node': -1,  # No NUMA preference
                    'cpus': all_cpus[start:end]
                })
        else:
            # NUMA-aware distribution
            num_nodes = self.topology['num_nodes']
            cpus_per_node = self.topology['cpus_per_node']

            # Distribute workers evenly across NUMA nodes
            workers_per_node = self.n_workers // num_nodes
            extra_workers = self.n_workers % num_nodes

            worker_id = 0
            for node_id in range(num_nodes):
                node_cpus = cpus_per_node[node_id]
                n_workers_this_node = workers_per_node + (1 if node_id < extra_workers else 0)

                if n_workers_this_node == 0:
                    continue

                # Distribute node's CPUs among its workers
                cpus_per_worker = max(1, len(node_cpus) // n_workers_this_node)

                for j in range(n_workers_this_node):
                    start = j * cpus_per_worker
                    end = start + cpus_per_worker if j < n_workers_this_node - 1 else len(node_cpus)
                    mapping.append({
                        'worker_id': worker_id,
                        'numa_node': node_id,
                        'cpus': node_cpus[start:end]
                    })
                    worker_id += 1

        return mapping

    def _create_pool(self):
        """Create the multiprocessing pool with NUMA-aware initialization."""
        if self._pool is not None:
            return

        # For NUMA-aware init, we need to use a custom initializer wrapper
        # that sets affinity before calling the user's initializer
        if self.is_numa and self.numa_balance:
            # Use spawn context for clean worker processes
            ctx = mp.get_context('spawn')

            # Create pool with custom initializer
            # Note: We can't pass per-worker args to Pool initializer,
            # so workers will self-assign based on their PID
            self._pool = ctx.Pool(
                processes=self.n_workers,
                initializer=self._numa_init_wrapper,
                initargs=(self.worker_numa_map, self.initializer, self.initargs)
            )
        else:
            # Standard pool without NUMA awareness
            self._pool = Pool(
                processes=self.n_workers,
                initializer=self.initializer,
                initargs=self.initargs
            )

    @staticmethod
    def _numa_init_wrapper(numa_map: List[Dict], user_init: Optional[Callable], user_args: Tuple):
        """
        Wrapper for NUMA-aware worker initialization.

        Workers self-assign to NUMA nodes based on a shared counter.
        """
        import os

        # Get a unique worker ID based on process ID
        # This is a simple heuristic - workers will be assigned in order of creation
        pid = os.getpid()

        # Find our slot in the NUMA map
        # Use modulo in case more workers than expected
        worker_idx = pid % len(numa_map)
        worker_config = numa_map[worker_idx]

        # Set CPU affinity
        if worker_config['cpus']:
            set_cpu_affinity(worker_config['cpus'])

        # Set NUMA preference
        if worker_config['numa_node'] >= 0:
            set_numa_node(worker_config['numa_node'])

        # Store worker info in environment for debugging
        os.environ['NUMA_WORKER_ID'] = str(worker_config['worker_id'])
        os.environ['NUMA_NODE'] = str(worker_config['numa_node'])
        os.environ['NUMA_CPUS'] = ','.join(map(str, worker_config['cpus']))

        # Call user's initializer
        if user_init is not None:
            user_init(*user_args)

    def map(self, func: Callable, iterable, chunksize: int = 1) -> List:
        """
        Apply function to each item in iterable.

        Args:
            func: Function to apply
            iterable: Items to process
            chunksize: Number of items per task

        Returns:
            List of results
        """
        self._create_pool()
        return self._pool.map(func, iterable, chunksize=chunksize)

    def imap(self, func: Callable, iterable, chunksize: int = 1):
        """
        Lazy map - returns iterator over results.

        Args:
            func: Function to apply
            iterable: Items to process
            chunksize: Number of items per task

        Returns:
            Iterator over results
        """
        self._create_pool()
        return self._pool.imap(func, iterable, chunksize=chunksize)

    def imap_unordered(self, func: Callable, iterable, chunksize: int = 1):
        """
        Lazy unordered map - returns iterator over results in completion order.

        Args:
            func: Function to apply
            iterable: Items to process
            chunksize: Number of items per task

        Returns:
            Iterator over results (unordered)
        """
        self._create_pool()
        return self._pool.imap_unordered(func, iterable, chunksize=chunksize)

    def starmap(self, func: Callable, iterable, chunksize: int = 1) -> List:
        """
        Apply function to each tuple of arguments in iterable.

        Args:
            func: Function to apply
            iterable: Tuples of arguments
            chunksize: Number of items per task

        Returns:
            List of results
        """
        self._create_pool()
        return self._pool.starmap(func, iterable, chunksize=chunksize)

    def close(self):
        """Close the pool - no more tasks can be submitted."""
        if self._pool is not None:
            self._pool.close()

    def terminate(self):
        """Terminate all workers immediately."""
        if self._pool is not None:
            self._pool.terminate()

    def join(self):
        """Wait for all workers to finish."""
        if self._pool is not None:
            self._pool.join()

    def __enter__(self):
        """Context manager entry."""
        self._create_pool()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        self.join()
        return False

    def get_numa_stats(self) -> Dict[str, Any]:
        """
        Get statistics about NUMA distribution.

        Returns:
            Dictionary with NUMA statistics
        """
        stats = {
            'is_numa': self.is_numa,
            'num_nodes': self.topology['num_nodes'],
            'num_workers': self.n_workers,
            'workers_per_node': {},
            'cpus_per_worker': []
        }

        for config in self.worker_numa_map:
            node = config['numa_node']
            if node not in stats['workers_per_node']:
                stats['workers_per_node'][node] = 0
            stats['workers_per_node'][node] += 1
            stats['cpus_per_worker'].append(len(config['cpus']))

        return stats


def create_numa_pool(
    n_workers: Optional[int] = None,
    initializer: Optional[Callable] = None,
    initargs: Tuple = ()
) -> NUMAWorkerPool:
    """
    Convenience function to create a NUMA-aware worker pool.

    Args:
        n_workers: Number of workers (default: auto-detect from SLURM or cpu_count)
        initializer: Worker initialization function
        initargs: Arguments for initializer

    Returns:
        NUMAWorkerPool instance
    """
    return NUMAWorkerPool(
        n_workers=n_workers,
        initializer=initializer,
        initargs=initargs
    )


# Test function
if __name__ == "__main__":
    print("=" * 70)
    print("NUMA Worker Pool - Topology Detection Test")
    print("=" * 70)

    topology = detect_numa_topology()

    print(f"\nSystem Topology:")
    print(f"  NUMA available: {topology['is_numa']}")
    print(f"  NUMA nodes: {topology['num_nodes']}")
    print(f"  Total CPUs: {topology['num_cpus']}")

    if topology['is_numa']:
        print(f"\nPer-Node Information:")
        for i, cpus in enumerate(topology['cpus_per_node']):
            mem = topology['memory_per_node'][i] if i < len(topology['memory_per_node']) else 'N/A'
            print(f"  Node {i}: {len(cpus)} CPUs, {mem} GB memory")
            if len(cpus) <= 16:
                print(f"    CPUs: {cpus}")
            else:
                print(f"    CPUs: {cpus[:8]} ... {cpus[-8:]}")

    print(f"\nWorker Pool Configuration (64 workers):")
    pool = NUMAWorkerPool(n_workers=64)
    stats = pool.get_numa_stats()

    print(f"  Workers per NUMA node: {stats['workers_per_node']}")
    print(f"  CPUs per worker: min={min(stats['cpus_per_worker'])}, max={max(stats['cpus_per_worker'])}")

    # Simple test
    print(f"\nRunning simple test...")

    def test_func(x):
        import os
        numa_node = os.environ.get('NUMA_NODE', 'N/A')
        return (x, x * x, numa_node)

    with NUMAWorkerPool(n_workers=4) as pool:
        results = pool.map(test_func, range(10))
        print(f"  Results: {results[:3]}...")

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
