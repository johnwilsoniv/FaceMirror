#!/usr/bin/env python3
"""
Comprehensive Performance Profiling Tool for Face Mirror

This module provides detailed timing instrumentation for all critical code paths:
- Model inference (RetinaFace, STAR, MTL)
- Memory operations (frame read, write, copy)
- Face mirroring pipeline
- AU extraction pipeline

Usage:
    from performance_profiler import PerformanceProfiler

    profiler = PerformanceProfiler()

    with profiler.time_block("model_inference", "RetinaFace"):
        result = model.detect_faces(frame)

    profiler.print_report()
"""

import time
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple
import os

# Optional: psutil for memory tracking (graceful degradation if not available)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class PerformanceProfiler:
    """
    Thread-safe performance profiler for detailed timing analysis.

    Tracks:
    - Total time per operation type
    - Count of operations
    - Min/max/avg timing
    - Memory usage deltas
    - Neural Engine vs CPU time (estimated)
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize performance profiler.

        Args:
            enabled: Whether profiling is active (default: True)
        """
        self.enabled = enabled
        self.lock = threading.Lock()

        # Timing data: {category: {operation: [times]}}
        self.timings: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

        # Memory data: {category: {operation: [memory_delta_mb]}}
        self.memory: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

        # Count data: {category: {operation: count}}
        self.counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Thread-local storage for nested timers
        self._thread_local = threading.local()

        # Process handle for memory tracking (if psutil available)
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process(os.getpid())
        else:
            self.process = None

    def reset(self):
        """Clear all profiling data."""
        with self.lock:
            self.timings.clear()
            self.memory.clear()
            self.counts.clear()

    @contextmanager
    def time_block(self, category: str, operation: str, track_memory: bool = False):
        """
        Context manager for timing a block of code.

        Args:
            category: Category name (e.g., "model_inference", "memory_ops")
            operation: Operation name (e.g., "RetinaFace", "frame_read")
            track_memory: Whether to track memory usage (default: False, adds ~1ms overhead)

        Example:
            with profiler.time_block("model_inference", "RetinaFace"):
                result = model.detect_faces(frame)
        """
        if not self.enabled:
            yield
            return

        # Record start state
        start_time = time.perf_counter()
        start_memory = None
        if track_memory and self.process is not None:
            start_memory = self.process.memory_info().rss / 1024**2

        try:
            yield
        finally:
            # Record end state
            elapsed = time.perf_counter() - start_time

            # Update timing data
            with self.lock:
                self.timings[category][operation].append(elapsed)
                self.counts[category][operation] += 1

                # Track memory if requested
                if track_memory and start_memory is not None and self.process is not None:
                    end_memory = self.process.memory_info().rss / 1024**2
                    memory_delta = end_memory - start_memory
                    self.memory[category][operation].append(memory_delta)

    def record_timing(self, category: str, operation: str, elapsed_seconds: float):
        """
        Manually record a timing measurement.

        Args:
            category: Category name
            operation: Operation name
            elapsed_seconds: Time elapsed in seconds
        """
        if not self.enabled:
            return

        with self.lock:
            self.timings[category][operation].append(elapsed_seconds)
            self.counts[category][operation] += 1

    def get_stats(self, category: str, operation: str) -> Dict[str, float]:
        """
        Get statistics for a specific operation.

        Args:
            category: Category name
            operation: Operation name

        Returns:
            Dictionary with min, max, avg, total, count, and percentage stats
        """
        if category not in self.timings or operation not in self.timings[category]:
            return {}

        times = self.timings[category][operation]
        if not times:
            return {}

        total_time = sum(times)
        count = len(times)

        # Calculate category total for percentage
        category_total = sum(
            sum(sum(times) for times in operations.values())
            for operations in self.timings.values()
        )

        return {
            'count': count,
            'total': total_time,
            'min': min(times),
            'max': max(times),
            'avg': total_time / count if count > 0 else 0,
            'percentage': (total_time / category_total * 100) if category_total > 0 else 0,
        }

    def print_report(self, detailed: bool = True):
        """
        Print comprehensive profiling report.

        Args:
            detailed: Whether to show detailed per-operation stats (default: True)
        """
        import sys

        if not self.enabled:
            print("Profiling is disabled", flush=True)
            return

        print("\n" + "="*80, flush=True)
        print("PERFORMANCE PROFILING REPORT", flush=True)
        print("="*80, flush=True)

        # Calculate grand total
        grand_total = sum(
            sum(sum(times) for times in operations.values())
            for operations in self.timings.values()
        )

        if grand_total == 0:
            print("No timing data collected", flush=True)
            print(f"  Categories present: {list(self.timings.keys())}", flush=True)
            print(f"  Profiler enabled: {self.enabled}", flush=True)
            return

        print(f"\nTotal Profiled Time: {grand_total:.3f}s", flush=True)
        print("", flush=True)

        # Sort categories by total time (descending)
        category_totals = {
            cat: sum(sum(times) for times in operations.values())
            for cat, operations in self.timings.items()
        }
        sorted_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)

        # Print each category
        for category, cat_total in sorted_categories:
            print(f"\n{'─'*80}", flush=True)
            print(f"Category: {category.upper()}", flush=True)
            print(f"{'─'*80}", flush=True)
            print(f"Total Time: {cat_total:.3f}s ({cat_total/grand_total*100:.1f}% of total)", flush=True)
            print("", flush=True)

            # Sort operations by total time (descending)
            operations = self.timings[category]
            op_totals = {op: sum(times) for op, times in operations.items()}
            sorted_ops = sorted(op_totals.items(), key=lambda x: x[1], reverse=True)

            if detailed:
                # Detailed view: show min/max/avg for each operation
                print(f"{'Operation':<30} {'Count':>8} {'Total':>10} {'Avg':>10} {'Min':>10} {'Max':>10} {'%':>6}", flush=True)
                print(f"{'-'*30} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*6}", flush=True)

                for operation, op_total in sorted_ops:
                    times = operations[operation]
                    count = len(times)
                    avg = op_total / count if count > 0 else 0
                    min_time = min(times) if times else 0
                    max_time = max(times) if times else 0
                    percentage = (op_total / cat_total * 100) if cat_total > 0 else 0

                    print(f"{operation:<30} {count:>8} {op_total:>9.3f}s "
                          f"{avg*1000:>9.1f}ms {min_time*1000:>9.1f}ms "
                          f"{max_time*1000:>9.1f}ms {percentage:>5.1f}%", flush=True)

                    # Show memory data if available
                    if category in self.memory and operation in self.memory[category]:
                        mem_deltas = self.memory[category][operation]
                        if mem_deltas:
                            avg_mem = sum(mem_deltas) / len(mem_deltas)
                            print(f"  └─ Memory: {avg_mem:>+.1f} MB avg per call", flush=True)
            else:
                # Summary view: just operation totals
                print(f"{'Operation':<40} {'Total':>12} {'%':>6}", flush=True)
                print(f"{'-'*40} {'-'*12} {'-'*6}", flush=True)

                for operation, op_total in sorted_ops:
                    percentage = (op_total / cat_total * 100) if cat_total > 0 else 0
                    print(f"{operation:<40} {op_total:>11.3f}s {percentage:>5.1f}%", flush=True)

        print("\n" + "="*80, flush=True)
        print("END OF REPORT", flush=True)
        print("="*80 + "\n", flush=True)

    def export_json(self, filepath: str):
        """
        Export profiling data to JSON file.

        Args:
            filepath: Path to output JSON file
        """
        import json

        data = {
            'timings': {
                cat: {op: list(times) for op, times in operations.items()}
                for cat, operations in self.timings.items()
            },
            'memory': {
                cat: {op: list(mem) for op, mem in operations.items()}
                for cat, operations in self.memory.items()
            },
            'counts': {
                cat: dict(operations) for cat, operations in self.counts.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Profiling data exported to: {filepath}")

    def get_summary_dict(self) -> Dict:
        """
        Get profiling summary as a dictionary.

        Returns:
            Dictionary with category summaries
        """
        summary = {}

        for category, operations in self.timings.items():
            cat_total = sum(sum(times) for times in operations.values())

            ops_summary = {}
            for operation, times in operations.items():
                if times:
                    ops_summary[operation] = {
                        'count': len(times),
                        'total': sum(times),
                        'avg': sum(times) / len(times),
                        'min': min(times),
                        'max': max(times),
                    }

            summary[category] = {
                'total': cat_total,
                'operations': ops_summary
            }

        return summary


# Global profiler instance (can be disabled by setting enabled=False)
_global_profiler = None


def get_profiler() -> PerformanceProfiler:
    """
    Get or create the global profiler instance.

    Profiling is controlled by config.ENABLE_PROFILING (default: False).

    Returns:
        Global PerformanceProfiler instance
    """
    global _global_profiler
    if _global_profiler is None:
        # Import config to check if profiling is enabled
        try:
            import config
            enabled = config.ENABLE_PROFILING
        except ImportError:
            # Fallback if config not available
            enabled = False

        _global_profiler = PerformanceProfiler(enabled=enabled)
    return _global_profiler


def set_profiler_enabled(enabled: bool):
    """
    Enable or disable the global profiler.

    Args:
        enabled: Whether profiling should be active
    """
    profiler = get_profiler()
    profiler.enabled = enabled


# Convenience functions for common timing operations
def time_model_inference(model_name: str):
    """
    Context manager for timing model inference.

    Example:
        with time_model_inference("RetinaFace"):
            result = model.detect_faces(frame)
    """
    return get_profiler().time_block("model_inference", model_name)


def time_memory_operation(operation_name: str):
    """
    Context manager for timing memory operations.

    Example:
        with time_memory_operation("frame_read"):
            ret, frame = cap.read()
    """
    return get_profiler().time_block("memory_operations", operation_name, track_memory=True)


def time_preprocessing(operation_name: str):
    """
    Context manager for timing preprocessing operations.

    Example:
        with time_preprocessing("image_resize"):
            resized = cv2.resize(image, (640, 480))
    """
    return get_profiler().time_block("preprocessing", operation_name)


if __name__ == '__main__':
    # Example usage and self-test
    print("Performance Profiler Self-Test")
    print("="*60)

    profiler = PerformanceProfiler()

    # Simulate some operations
    print("Simulating operations...")

    for i in range(10):
        with profiler.time_block("model_inference", "RetinaFace"):
            time.sleep(0.01)  # Simulate 10ms inference

        with profiler.time_block("model_inference", "STAR"):
            time.sleep(0.015)  # Simulate 15ms inference

        with profiler.time_block("memory_operations", "frame_read"):
            time.sleep(0.002)  # Simulate 2ms read

        with profiler.time_block("preprocessing", "image_resize"):
            time.sleep(0.003)  # Simulate 3ms resize

    # Print report
    profiler.print_report(detailed=True)

    # Test JSON export
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        profiler.export_json(f.name)
        print(f"\nJSON export test: {f.name}")
