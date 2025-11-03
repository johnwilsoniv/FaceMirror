"""
Enhanced Diagnostic Profiler
Provides detailed component-level timing and statistics for bottleneck identification
"""

import time
from collections import defaultdict, deque
from datetime import datetime
import json


class DiagnosticProfiler:
    """
    Detailed profiling for specific operations with component-level breakdown
    """

    def __init__(self):
        self.enabled = True

        # Component timing storage
        self.timings = defaultdict(lambda: deque(maxlen=100))  # Keep last 100 samples per component

        # Cache statistics
        self.cache_stats = {
            'rgb_hits': 0,
            'rgb_misses': 0,
            'qimage_hits': 0,
            'qimage_misses': 0,
            'total_cache_time_saved_ms': 0.0
        }

        # Paint event breakdown
        self.paint_components = defaultdict(lambda: deque(maxlen=50))

        # Frame extraction breakdown
        self.frame_extraction = defaultdict(lambda: deque(maxlen=100))

        # Memory snapshots
        self.memory_snapshots = deque(maxlen=100)

        # Operation counters
        self.operation_counts = defaultdict(int)

    def time_operation(self, category, operation_name):
        """
        Context manager for timing operations

        Usage:
            with profiler.time_operation('video', 'frame_seek'):
                # code to time
                pass
        """
        return _TimingContext(self, category, operation_name)

    def record_timing(self, category, operation_name, duration_ms):
        """Record a timing measurement"""
        if not self.enabled:
            return

        key = f"{category}.{operation_name}"
        self.timings[key].append(duration_ms)
        self.operation_counts[key] += 1

    def record_cache_hit(self, cache_type, time_saved_ms=0):
        """Record cache hit"""
        if cache_type == 'rgb':
            self.cache_stats['rgb_hits'] += 1
        elif cache_type == 'qimage':
            self.cache_stats['qimage_hits'] += 1
        self.cache_stats['total_cache_time_saved_ms'] += time_saved_ms

    def record_cache_miss(self, cache_type):
        """Record cache miss"""
        if cache_type == 'rgb':
            self.cache_stats['rgb_misses'] += 1
        elif cache_type == 'qimage':
            self.cache_stats['qimage_misses'] += 1

    def record_paint_component(self, component_name, duration_ms):
        """Record paint event component timing"""
        if not self.enabled:
            return
        self.paint_components[component_name].append(duration_ms)

    def record_memory_snapshot(self, operation, rss_mb):
        """Record memory usage at a specific point"""
        if not self.enabled:
            return
        self.memory_snapshots.append({
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'rss_mb': rss_mb
        })

    def get_statistics(self):
        """Generate comprehensive statistics"""
        stats = {
            'timing_breakdown': {},
            'cache_performance': self._get_cache_stats(),
            'paint_breakdown': self._get_paint_stats(),
            'frame_extraction_breakdown': self._get_frame_extraction_stats(),
            'top_bottlenecks': self._get_top_bottlenecks(),
            'operation_counts': dict(self.operation_counts)
        }

        # Timing breakdown
        for key, samples in self.timings.items():
            if samples:
                samples_list = list(samples)
                stats['timing_breakdown'][key] = {
                    'count': len(samples_list),
                    'avg_ms': sum(samples_list) / len(samples_list),
                    'min_ms': min(samples_list),
                    'max_ms': max(samples_list),
                    'total_ms': sum(samples_list)
                }

        return stats

    def _get_cache_stats(self):
        """Calculate cache performance metrics"""
        total_rgb = self.cache_stats['rgb_hits'] + self.cache_stats['rgb_misses']
        total_qimage = self.cache_stats['qimage_hits'] + self.cache_stats['qimage_misses']

        rgb_hit_rate = (self.cache_stats['rgb_hits'] / total_rgb * 100) if total_rgb > 0 else 0
        qimage_hit_rate = (self.cache_stats['qimage_hits'] / total_qimage * 100) if total_qimage > 0 else 0

        return {
            'rgb_cache': {
                'hits': self.cache_stats['rgb_hits'],
                'misses': self.cache_stats['rgb_misses'],
                'hit_rate_percent': round(rgb_hit_rate, 2),
                'total_requests': total_rgb
            },
            'qimage_cache': {
                'hits': self.cache_stats['qimage_hits'],
                'misses': self.cache_stats['qimage_misses'],
                'hit_rate_percent': round(qimage_hit_rate, 2),
                'total_requests': total_qimage
            },
            'total_time_saved_ms': round(self.cache_stats['total_cache_time_saved_ms'], 2)
        }

    def _get_paint_stats(self):
        """Get paint event component statistics"""
        stats = {}
        for component, samples in self.paint_components.items():
            if samples:
                samples_list = list(samples)
                stats[component] = {
                    'avg_ms': round(sum(samples_list) / len(samples_list), 2),
                    'max_ms': round(max(samples_list), 2),
                    'count': len(samples_list)
                }
        return stats

    def _get_frame_extraction_stats(self):
        """Get frame extraction breakdown"""
        stats = {}
        for operation, samples in self.frame_extraction.items():
            if samples:
                samples_list = list(samples)
                stats[operation] = {
                    'avg_ms': round(sum(samples_list) / len(samples_list), 2),
                    'max_ms': round(max(samples_list), 2),
                    'count': len(samples_list)
                }
        return stats

    def _get_top_bottlenecks(self, top_n=10):
        """Identify top bottlenecks by total time spent"""
        bottlenecks = []

        for key, samples in self.timings.items():
            if samples:
                samples_list = list(samples)
                total_time = sum(samples_list)
                avg_time = total_time / len(samples_list)
                bottlenecks.append({
                    'operation': key,
                    'total_ms': round(total_time, 2),
                    'avg_ms': round(avg_time, 2),
                    'count': len(samples_list),
                    'max_ms': round(max(samples_list), 2)
                })

        # Sort by total time descending
        bottlenecks.sort(key=lambda x: x['total_ms'], reverse=True)
        return bottlenecks[:top_n]

    def print_summary(self):
        """Print a human-readable summary"""
        stats = self.get_statistics()

        print("\n" + "="*70)
        print("DIAGNOSTIC PROFILER SUMMARY")
        print("="*70)

        # Cache performance
        print("\nCACHE PERFORMANCE:")
        cache = stats['cache_performance']
        print(f"  RGB Cache:    {cache['rgb_cache']['hit_rate_percent']:.1f}% hit rate "
              f"({cache['rgb_cache']['hits']} hits / {cache['rgb_cache']['total_requests']} total)")
        print(f"  QImage Cache: {cache['qimage_cache']['hit_rate_percent']:.1f}% hit rate "
              f"({cache['qimage_cache']['hits']} hits / {cache['qimage_cache']['total_requests']} total)")
        print(f"  Time Saved:   {cache['total_time_saved_ms']:.1f}ms total")

        # Top bottlenecks
        print("\n🔥 TOP BOTTLENECKS (by total time):")
        for i, bottleneck in enumerate(stats['top_bottlenecks'], 1):
            print(f"  {i}. {bottleneck['operation']}")
            print(f"     Total: {bottleneck['total_ms']:.1f}ms | "
                  f"Avg: {bottleneck['avg_ms']:.2f}ms | "
                  f"Max: {bottleneck['max_ms']:.2f}ms | "
                  f"Count: {bottleneck['count']}")

        # Paint breakdown
        if stats['paint_breakdown']:
            print("\n🎨 PAINT EVENT BREAKDOWN:")
            for component, data in stats['paint_breakdown'].items():
                print(f"  {component}: avg={data['avg_ms']:.2f}ms, max={data['max_ms']:.2f}ms")

        # Frame extraction
        if stats['frame_extraction_breakdown']:
            print("\n🎬 FRAME EXTRACTION BREAKDOWN:")
            for operation, data in stats['frame_extraction_breakdown'].items():
                print(f"  {operation}: avg={data['avg_ms']:.2f}ms, max={data['max_ms']:.2f}ms")

        print("\n" + "="*70)

    def save_report(self, filename=None):
        """Save detailed report to JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"diagnostic_report_{timestamp}.json"

        stats = self.get_statistics()

        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n📝 Diagnostic report saved to: {filename}")
        return filename


class _TimingContext:
    """Context manager for timing operations"""

    def __init__(self, profiler, category, operation_name):
        self.profiler = profiler
        self.category = category
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        self.profiler.record_timing(self.category, self.operation_name, duration_ms)
        return False  # Don't suppress exceptions


# Global instance
_diagnostic_profiler = None

def get_diagnostic_profiler():
    """Get or create the global diagnostic profiler instance"""
    global _diagnostic_profiler
    if _diagnostic_profiler is None:
        _diagnostic_profiler = DiagnosticProfiler()
    return _diagnostic_profiler
