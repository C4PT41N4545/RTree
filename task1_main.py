from my_rtree import RTree
import time
from pathlib import Path
from utils import read_point, write_summary, divide_data
from task1_sequintial_search import sequential_search
from task1_best_first_search import best_first_search
from task1_divide_best_first_search import best_first_divide_search
from visualisation import plot_grouped_performance
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
    parser.add_argument('--results-dir', type=Path, default=Path('Task1_Results'), help='Directory to save the results.')
    args = parser.parse_args()
    MAX_ENTRIES = args.max_entries
    dataset_dir = args.dataset_dir
    query_dir = args.query_dir
    results_dir = args.results_dir
    dataset_name = dataset_dir.name.replace('.txt', '')
    # Read the dataset and query files
    data_file  = dataset_dir
    query_file = query_dir
    # Read the dataset
    data_points = read_point(data_file)
    # Read the query points
    query_points = read_point(query_file)
    #construct the RTree
    start_time = time.time()
    rtree = RTree(max_entries=MAX_ENTRIES)
    rtree.fit(data_points)
    end_time = time.time()
    pre_processing_time = end_time - start_time
    # construct the RTree for the divided data
    start_time = time.time()
    data1, data2 = divide_data(data_points)
    rtree1 = RTree(max_entries=MAX_ENTRIES)
    rtree1.fit(data1)
    rtree2 = RTree(max_entries=MAX_ENTRIES)
    rtree2.fit(data2)
    end_time = time.time()
    pre_processing_time_divide = end_time - start_time
    all_results = {}
    # Perform sequential search
    results1, average_time1, total_time1 = sequential_search(data_points, query_points)
    all_results['Sequential Search'] = {'results': results1, 'average_time': average_time1, 'total_time': total_time1}
    # Perform best-first search
    results2, average_time2, total_time2 = best_first_search(rtree, query_points)
    all_results['Best First Search'] = {'results': results2, 'average_time': average_time2, 'total_time': total_time2}
    #Perform best-first divide search
    results3, average_time3, total_time3 = best_first_divide_search(rtree1, rtree2, query_points)
    all_results['Best First Divide Search'] = {'results': results3, 'average_time': average_time3, 'total_time': total_time3}
    print(f"Total queries: {len(query_points)}")
    # Cross-validate the results
    incorrect_count = cross_validate(results1, results2, results3)
    print(f"Cross-validation found {incorrect_count} discrepancies among the three methods.")
    # Save all results to a summary file
    write_summary(all_results, results_dir, dataset_name=dataset_name)
    print(f"Results written to {results_dir.resolve()}")
    # Plot the performance of the algorithms
    all_results['Pre-processing Time'] = {
        'Sequential Search': 0,  # Sequential search does not have pre-processing time
        'Best First Search': pre_processing_time,
        'Best First Divide Search': pre_processing_time_divide
    }
    all_results['Average Time'] = {
        'Sequential Search': average_time1,
        'Best First Search': average_time2,
        'Best First Divide Search': average_time3
    }
    all_results['Total Time'] = {
        'Sequential Search': total_time1,
        'Best First Search': total_time2,
        'Best First Divide Search': total_time3
    }
    plot_grouped_performance(all_results, results_dir, dataset_name=dataset_name)
if __name__ == "__main__":
    main()