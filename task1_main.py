from my_rtree import RTree, Node
import sys
import math
import numpy as np
import time
import os
from pathlib import Path
from utils import read_point, write_summary, divide_data
from task1_sequintial_search import sequential_search
from task1_best_first_search import best_first_search
from task1_divide_best_first_search import best_first_divide_search
import argparse
# --------------------------------------------------------------------------- #
# Main orchestration
# --------------------------------------------------------------------------- #
def cross_validate(results1, results2, results3) -> int:
    """Cross-validate the results of three search methods."""
    incorrect_count = 0
    for i in range(len(results1)):
        if results1[i] != results2[i] or results1[i] != results3[i]:
            incorrect_count += 1
    return incorrect_count
def main() -> None:
    '''Main orchestration function to run the search algorithms and save results.'''
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run nearest neighbor search algorithms.")
    parser.add_argument('--max_entries', type=int, default=10, help='Maximum entries per R-tree node.')
    parser.add_argument('--dataset-dir', type=Path, default=Path('Task1_Datasets/parking_dataset.txt'), help='Directory containing the dataset.')
    parser.add_argument('--query-dir', type=Path, default=Path('Task1_Datasets/query_points.txt'), help='Directory containing the dataset.')
    parser.add_argument('--results-dir', type=Path, default=Path('Task1_Results/all_results.txt'), help='Directory to save the results.')
    args = parser.parse_args()
    MAX_ENTRIES = args.max_entries
    dataset_dir = args.dataset_dir
    query_dir = args.query_dir
    results_dir = args.results_dir
 
    # Read the dataset and query files
    data_file  = dataset_dir
    query_file = query_dir
    # Read the dataset
    data_points = read_point(data_file)
    # Read the query points
    query_points = read_point(query_file)
    #construct the RTree
    rtree = RTree(max_entries=MAX_ENTRIES)
    rtree.fit(data_points)
    # construct the RTree for the divided data
    data1, data2 = divide_data(data_points)
    rtree1 = RTree(max_entries=MAX_ENTRIES)
    rtree1.fit(data1)
    rtree2 = RTree(max_entries=MAX_ENTRIES)
    rtree2.fit(data2)
    all_results = {}
    # Perform sequential search
    results1, average_time, total_time = sequential_search(data_points, query_points)
    all_results['Sequential Search'] = {'results': results1, 'average_time': average_time, 'total_time': total_time}
    # Perform best-first search
    results2, average_time, total_time = best_first_search(rtree, query_points)
    all_results['Best First Search'] = {'results': results2, 'average_time': average_time, 'total_time': total_time}
    #Perform best-first divide search
    results3, average_time, total_time = best_first_divide_search(rtree1, rtree2, query_points)
    all_results['Best First Divide Search'] = {'results': results3, 'average_time': average_time, 'total_time': total_time}
    print(f"Total queries: {len(query_points)}")
    # Cross-validate the results
    incorrect_count = cross_validate(results1, results2, results3)
    print(f"Cross-validation found {incorrect_count} discrepancies among the three methods.")
    # Save all results to a summary file
    write_summary(all_results, results_dir, dataset_name=dataset_dir.name)
    print(f"Results written to {results_dir.resolve()}")
if __name__ == "__main__":
    main()