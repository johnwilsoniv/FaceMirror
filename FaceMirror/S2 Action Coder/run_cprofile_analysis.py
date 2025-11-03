#!/usr/bin/env python3
"""
cProfile Analysis Wrapper
Runs Action Coder with Python's cProfile to identify hot spots and function-level bottlenecks
"""

import cProfile
import pstats
import io
import sys
import os
from datetime import datetime


def run_with_profiler():
    """Run Action Coder with cProfile enabled"""

    print("="*70)
    print("cProfile Analysis - Function-Level Profiling")
    print("="*70)
    print("Starting Action Coder with cProfile instrumentation...")
    print("This will profile ALL function calls to identify hot spots.")
    print("")
    print("Instructions:")
    print("1. Load a video file")
    print("2. Play for 10-15 seconds")
    print("3. Scrub the timeline (click to seek)")
    print("4. Close the application")
    print("")
    print("Analysis will be saved automatically on exit.")
    print("="*70)
    print("")

    # Create profiler
    profiler = cProfile.Profile()

    # Start profiling
    profiler.enable()

    try:
        # Import and run the main application
        # This needs to be imported AFTER profiler.enable() to capture everything
        from main import main
        main()

    except Exception as e:
        print(f"\nError during profiling: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop profiling
        profiler.disable()

        # Generate timestamp for report files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw profile data
        stats_file = f"cprofile_stats_{timestamp}.prof"
        profiler.dump_stats(stats_file)
        print(f"\n✓ Raw profile data saved to: {stats_file}")

        # Generate human-readable reports
        generate_reports(profiler, timestamp)


def generate_reports(profiler, timestamp):
    """Generate human-readable analysis reports from cProfile data"""

    # Create a Stats object
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)

    # Report 1: Top functions by cumulative time (shows bottlenecks)
    report1_file = f"cprofile_cumulative_{timestamp}.txt"
    with open(report1_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TOP BOTTLENECKS - Functions by Cumulative Time\n")
        f.write("="*70 + "\n")
        f.write("Cumulative time = time spent in function + all functions it calls\n")
        f.write("This shows where the application spends the MOST total time.\n")
        f.write("="*70 + "\n\n")

        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats(50)  # Top 50 functions
        f.write(stream.getvalue())

    print(f"✓ Cumulative time report saved to: {report1_file}")

    # Report 2: Top functions by internal time (pure function time)
    report2_file = f"cprofile_tottime_{timestamp}.txt"
    with open(report2_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PURE FUNCTION TIME - Functions by Internal Time Only\n")
        f.write("="*70 + "\n")
        f.write("Internal time = time spent in function itself (excluding called functions)\n")
        f.write("This shows which individual operations are slowest.\n")
        f.write("="*70 + "\n\n")

        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('tottime')
        stats.print_stats(50)  # Top 50 functions
        f.write(stream.getvalue())

    print(f"✓ Internal time report saved to: {report2_file}")

    # Report 3: Video player specific functions
    report3_file = f"cprofile_video_player_{timestamp}.txt"
    with open(report3_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("VIDEO PLAYER ANALYSIS - qt_media_player.py Functions\n")
        f.write("="*70 + "\n")
        f.write("Filtering for video frame extraction and playback functions.\n")
        f.write("="*70 + "\n\n")

        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats('qt_media_player')  # Filter for video player functions
        f.write(stream.getvalue())

    print(f"✓ Video player analysis saved to: {report3_file}")

    # Report 4: Timeline widget specific functions
    report4_file = f"cprofile_timeline_{timestamp}.txt"
    with open(report4_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TIMELINE WIDGET ANALYSIS - timeline_widget.py Functions\n")
        f.write("="*70 + "\n")
        f.write("Filtering for timeline rendering and event handling functions.\n")
        f.write("="*70 + "\n\n")

        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats('timeline_widget')  # Filter for timeline functions
        f.write(stream.getvalue())

    print(f"✓ Timeline analysis saved to: {report4_file}")

    # Report 5: Call counts (most frequently called functions)
    report5_file = f"cprofile_call_counts_{timestamp}.txt"
    with open(report5_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CALL FREQUENCY - Most Frequently Called Functions\n")
        f.write("="*70 + "\n")
        f.write("Functions called most often (may indicate optimization opportunities).\n")
        f.write("="*70 + "\n\n")

        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('ncalls')
        stats.print_stats(50)  # Top 50 most called
        f.write(stream.getvalue())

    print(f"✓ Call frequency report saved to: {report5_file}")

    # Print summary to console
    print("\n" + "="*70)
    print("QUICK SUMMARY - Top 10 Bottlenecks by Cumulative Time")
    print("="*70)
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
    print(stream.getvalue())
    print("="*70)
    print("\n✓ All cProfile reports generated successfully!")
    print("\nNext steps:")
    print("1. Review cprofile_cumulative_*.txt for overall bottlenecks")
    print("2. Check cprofile_video_player_*.txt for frame extraction issues")
    print("3. Check cprofile_timeline_*.txt for UI rendering issues")
    print("4. Compare with diagnostic profiler results for complete picture")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_with_profiler()
