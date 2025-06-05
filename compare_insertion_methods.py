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
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - width / 2 for i in x], build_times, width, label="Build Time")
    ax.bar([i + width / 2 for i in x], query_times, width, label="Query Time")

    ax.set_ylabel("Time (seconds)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods)
    ax.legend()
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
