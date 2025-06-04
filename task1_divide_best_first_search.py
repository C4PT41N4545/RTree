from utils import read_point, write_file, divide_data
from my_rtree import RTree, Point
from tqdm import tqdm
import time
import argparse
from pathlib import Path
def best_first_divide_search(rtree1, rtree2, query_points):
    """Perform a best-first divide search using two R-trees for nearest neighbor."""
    total_start_time = time.time()
    results = []
    
    # Perform nearest neighbor search for each query point
    for query in tqdm(query_points, desc="Searching for nearest neighbors"):
        best_dist1, best_point1 = rtree1.nearest_neighbor(query)
        best_dist2, best_point2 = rtree2.nearest_neighbor(query)
        
        # Choose the best point from both halves
        if best_dist1 < best_dist2:
            best_point = best_point1
        else:
            best_point = best_point2
        
        results.append((best_point.id, best_point.x, best_point.y, query.id))
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    average_time = total_time / len(query_points)
    
    print(f"Divide Best First Search Time: {total_time:.2f} seconds")
    return results, average_time, total_time

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run nearest neighbor search algorithms.")
    parser.add_argument('--max_entries', type=int, default=10, help='Maximum entries per R-tree node.')
    parser.add_argument('--dataset-dir', type=Path, default=Path('Task1_Datasets/parking_dataset.txt'), help='Directory containing the dataset.')
    parser.add_argument('--query-dir', type=Path, default=Path('Task1_Datasets/query_points.txt'), help='Directory containing the dataset.')
    parser.add_argument('--results-dir', type=Path, default=Path('Task1_Results/divide_best_first_search_results.txt'), help='Directory to save the results.')
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
    
    # Divide data points into two halves
    data_points1, data_points2 = divide_data(data_points)
    #create an R-Tree for nearest neighbor search
    rtree1 = RTree(max_entries=MAX_ENTRIES)
    rtree1.fit(data_points1)
    # create an R-Tree for nearest neighbor search
    rtree2 = RTree(max_entries=MAX_ENTRIES)
    rtree2.fit(data_points2)
    results, average_time, total_time = best_first_divide_search(rtree1, rtree2, query_points)
    # Write results to file
    # Use the dataset name for the summary header rather than the results file
    # name to keep output consistent with other scripts.
    write_file(
        results_dir,
        results,
        average_time,
        total_time,
        "divide_best_first_search",
        dataset_dir.name,
    )
if __name__ == "__main__":
    main()