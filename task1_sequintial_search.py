import time
from utils import read_point, write_file, euclidean_distance
from tqdm import tqdm
from my_rtree import Point
import argparse
from pathlib import Path
def sequential_search(data_points: list[Point], query_points: list[Point]) -> tuple[list[str], float, float]:
    """Perform a sequential search for the nearest neighbor."""
    total_start_time = time.time()
    results: list[str] = []
    query_time: list[float] = []
    print("Beginning sequential search...")
    for query in tqdm(query_points, desc="Searching for nearest neighbors"):
        nearest_distance = float('inf')
        query_start_time = time.time()
        for data in data_points:
            # Calculate the distance
            distance = euclidean_distance(query, data)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_data = data
        results.append((nearest_data.id, nearest_data.x, nearest_data.y, query.id))
        query_time.append(time.time() - query_start_time)
    total_end_time = time.time()
    total_time: float = total_end_time - total_start_time
    average_time: float = total_time / len(query_points)
    print(f"Sequential Search Time: {total_time:.2f} seconds")
    return results, average_time, total_time
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run nearest neighbor search algorithms.")
    parser.add_argument('--dataset-dir', type=Path, default=Path('Task1_Datasets/parking_dataset.txt'), help='Directory containing the dataset.')
    parser.add_argument('--query-dir', type=Path, default=Path('Task1_Datasets/query_points.txt'), help='Directory containing the dataset.')
    parser.add_argument('--results-dir', type=Path, default=Path('Task1_Results/sequential_search_results.txt'), help='Directory to save the results.')
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    query_dir = args.query_dir
    results_dir = args.results_dir
    # Read data points from file
    data_points = read_point(dataset_dir)
    print(f"Total data points: {len(data_points)}")
    #Read query points from file
    query_points = read_point(query_dir)
    print(f"Total query points: {len(query_points)}")
    # Perform sequential search
    results, average_time, total_time = sequential_search(data_points, query_points)
    print(f"Total queries: {len(query_points)}")
    # Write results to file
    write_file(results_dir, results, average_time, total_time, "sequential", dataset_dir.name)
if __name__ == "__main__":
    main()