import time
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from my_rtree import RTree
from utils import read_point


def build_incremental(data_points, query_points, max_entries):
    """Build RTree incrementally and measure times."""
    rtree = RTree(max_entries=max_entries)
    start = time.time()
    rtree.fit(data_points, show_progress=False)
    build_time = time.time() - start

    start = time.time()
    rtree.search(query_points, show_progress=False)
    query_time = time.time() - start
    return build_time, query_time


def build_bulk(data_points, query_points, max_entries):
    """Build RTree using bulk_load and measure times."""
    rtree = RTree(max_entries=max_entries)
    start = time.time()
    rtree.bulk_load(data_points)
    build_time = time.time() - start

    start = time.time()
    rtree.search(query_points, show_progress=False)
    query_time = time.time() - start
    return build_time, query_time


def plot_results(build_times, query_times, results_dir):
    methods = ["Incremental", "Bulk Load"]
    x = range(len(methods))
    width = 0.6

    fig, (ax_build, ax_query) = plt.subplots(1, 2, figsize=(9, 4))

    build_bars = ax_build.bar(x, build_times, width, color="#8ecae6")
    query_bars = ax_query.bar(x, query_times, width, color="#219ebc")

    # Add value labels on top of the bars
    for bars in [build_bars, query_bars]:
        for bar in bars:
            height = bar.get_height()
            ax = bar.axes
            ax.annotate(f"{height:.6f}s",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    ax_build.set_ylabel("Time (seconds)")
    ax_build.set_title("Build Time")
    ax_query.set_title("Query Time")

    for axis in (ax_build, ax_query):
        axis.set_xticks(list(x))
        axis.set_xticklabels(methods)

    plt.tight_layout()
    plt.savefig(Path(results_dir) / "insertion_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Compare incremental insertion and bulk load.")
    parser.add_argument('--max-entries', type=int, default=10, help='Maximum entries per node.')
    parser.add_argument('--dataset-dir', type=Path, default=Path('Task1_Datasets/parking_dataset_sample.txt'))
    parser.add_argument('--query-dir', type=Path, default=Path('Task1_Datasets/query_points.txt'))
    parser.add_argument('--results-dir', type=Path, default=Path('Task1_Results'))
    args = parser.parse_args()

    data_points = read_point(args.dataset_dir)
    query_points = read_point(args.query_dir)

    inc_build, inc_query = build_incremental(data_points, query_points, args.max_entries)
    bulk_build, bulk_query = build_bulk(data_points, query_points, args.max_entries)

    print(f"Incremental build time: {inc_build:.6f}s, query time: {inc_query:.6f}s")
    print(f"Bulk load build time: {bulk_build:.6f}s, query time: {bulk_query:.6f}s")

    plot_results([inc_build, bulk_build], [inc_query, bulk_query], args.results_dir)


if __name__ == '__main__':
    main()
