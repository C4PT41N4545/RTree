from utils import read_point, write_file, write_all_results_to_file
from my_rtree import RTree
from tqdm import tqdm
import time
import argparse
from pathlib import Path
def best_first_search(rtree: RTree, query_points: list[dict[str, float]]) -> tuple[list[str], float, float]:
    """Perform a best-first search using the R-tree for nearest neighbor."""
    total_start_time = time.time()
    results: list[str] = []
    
    # Perform nearest neighbor search for each query point
    for query in tqdm(query_points, desc="Searching for nearest neighbors"):
        best_dist, best_point = rtree.nearest_neighbor(query)
        results.append(f"id={best_point.id}, x={best_point.x}, y={best_point.y} for query {query.id}")
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    average_time = total_time / len(query_points)
    
    print(f"Best First Search Time: {total_time:.4f} seconds")
    return results, average_time, total_time
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run nearest neighbor search algorithms.")
    parser.add_argument('--max_entries', type=int, default=10, help='Maximum entries per R-tree node.')
    parser.add_argument('--dataset-dir', type=Path, default=Path('Task1_Datasets/parking_dataset.txt'), help='Directory containing the dataset.')
    parser.add_argument('--query-dir', type=Path, default=Path('Task1_Datasets/query_points.txt'), help='Directory containing the dataset.')
    parser.add_argument('--results-dir', type=Path, default=Path('Task1_Results/best_first_search_results.txt'), help='Directory to save the results.')
    args = parser.parse_args()
    MAX_ENTRIES = args.max_entries
    dataset_dir = args.dataset_dir
    query_dir = args.query_dir
    results_dir = args.results_dir
    # Read data points from file
    data_points = read_point(dataset_dir)
    print(f"Total data points: {len(data_points)}")
    
    # Read query points from file
    query_points = read_point(query_dir)
    print(f"Total query points: {len(query_points)}")
    
    #create an R-Tree for nearest neighbor search
    rtree = RTree(max_entries=MAX_ENTRIES)
    rtree.fit(data_points)
    
    # Perform best-first search
    results, average_time, total_time = best_first_search(rtree, query_points)
    
    # Write results to file
    write_file(results_dir, results, average_time, total_time, "best_first_search", dataset_dir.name)
if __name__ == "__main__":
    main()