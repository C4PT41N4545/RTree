import math
import numpy as np
import time
from my_rtree import RTree, Node, Point
from pathlib import Path
# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def read_point(filename: str) -> list[Point]:
    """Read space-separated points from *filename*."""
    data_points: list = []
    with open(filename, 'r') as file:
        for line in file.readlines():
            line: list = line.split()
            id = int(line[0])
            x = float(line[1])
            y = float(line[2])
            data_points.append(Point(id=id, x=x, y=y))
    return data_points

def write_summary(results: dict[dict], filename: Path, dataset_name: str) -> None:
    '''Write all results to a specified file.'''
    filename = filename / f"summary_{dataset_name}.txt"
    print(f"Writing summary to {filename}")
    with open(filename, 'w') as f:
        f.write(f"Summary of Nearest Neighbor Search Results for {dataset_name}\n")
        for method in results:
            f.write(f"Method: {method}\n")
            f.write(f"Total Running Time: {results[method]['total_time']:.4f} seconds\n")
            f.write(f"Average Time per Query: {results[method]['average_time']:.4f} seconds\n")
            for res in results[method]['results']:
                res = f"id={res[0]}, x={res[1]}, y={res[2]} for query {res[3]}"
                f.write(f"{res}\n")
            f.write("-" * 80 + "\n")
    print(f"All results saved to {filename}")

def write_file(filename, results, average_time, total_time, name, dataset_name):
    #save the results to a file
    with open(filename, 'w') as result_file:
        result_file.write(
            f"Summary of Nearest Neighbor Search Results for {dataset_name}\n"
        )
        result_file.write(
            f"Time taken for nearest neighbor {name} search: {total_time:.4f} seconds\n"
        )
        # average_time already represents the per-query time, so do not divide
        # by the number of results again.
        result_file.write(
            f"Average time per query: {average_time:.6f} seconds\n"
        )
        for result in results:
            result = f"id={result[0]}, x={result[1]}, y={result[2]} for query {result[3]}"
            result_file.write(result + '\n')

def write_all_results_to_file(results, filename):
    with open(filename, 'w') as f:
        for method in results:
            f.write(f"Method: {method}\n")
            f.write(f"Total Running Time: {results[method]['total_time']:.4f} seconds\n")
            f.write(f"Average Time per Query: {results[method]['average_time']:.4f} seconds\n")
            for res in results[method]['results']:
                res = f"id={res[0]}, x={res[1]}, y={res[2]} for query {res[3]}"
                f.write(f"{res}\n")
            f.write("----------------------------------------------------------------------------------------\n")
    print(f"All results saved to {filename}")
# --------------------------------------------------------------------------- #
# Generic utilities
# --------------------------------------------------------------------------- #
def euclidean_distance(p1: Point, p2: Point) -> float:
    """Return Euclidean distance between two 2‑D points."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def divide_data(data_points: list[Point]) -> tuple[list[Point], list[Point]]:
    """Divide data points into two sets based on the median x-coordinate."""
    xs = [p.x for p in data_points]
    median_x = np.median(xs)
    data1: list[Point] = [p for p in data_points if p.x <= median_x]
    data2: list[Point] = [p for p in data_points if p.x > median_x]
    return data1, data2
