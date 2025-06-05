# Assignment 2: R-Tree Nearest Neighbor Search

This project implements and compares different nearest neighbor search algorithms using R-Trees for spatial data. It includes sequential search, best-first search, and a divide-and-conquer best-first search, along with visualization tools for performance analysis and R-Tree structure.

## Project Structure

- `my_rtree.py`: Core implementation of the R-Tree, including insertion, splitting, and nearest neighbor search.
- `utils.py`: Utility functions for reading/writing data, distance calculations, and data division.
- `task1_sequintial_search.py`: Sequential (brute-force) nearest neighbor search.
- `task1_best_first_search.py`: R-Tree best-first nearest neighbor search.
- `task1_divide_best_first_search.py`: Divide-and-conquer best-first search using two R-Trees.
- `task1_main.py`: Run all search algorithm and save result.
- `visualisation.py`: Visualization of R-Tree structure and performance metrics.
- `Task1_Datasets/`: Contains sample datasets and query points.
- `Task1_Results/`: Output results and plots are saved here.

## Requirements

- Python 3.x
- Required packages: `numpy`, `matplotlib`, `tqdm`

Install dependencies with:

```
pip install numpy matplotlib tqdm
```

## How to Run

### 1. Sequential Search

```
python task1_sequintial_search.py --dataset-dir Task1_Datasets/parking_dataset.txt --query-dir Task1_Datasets/query_points.txt --results-dir Task1_Results/sequential_search_results.txt
```

### 2. R-Tree Best-First Search

```
python task1_best_first_search.py --max_entries 10 --dataset-dir Task1_Datasets/parking_dataset.txt --query-dir Task1_Datasets/query_points.txt --results-dir Task1_Results/best_first_search_results.txt
```

### 3. Divide Best-First Search

```
python task1_divide_best_first_search.py --max_entries 10 --dataset-dir Task1_Datasets/parking_dataset.txt --query-dir Task1_Datasets/query_points.txt --results-dir Task1_Results/divide_best_first_search_results.txt
```
### 4. main

```
python task1_main.py --max_entries 10 --dataset-dir Task1_Datasets/parking_dataset.txt --query-dir Task1_Datasets/query_points.txt --results-dir Task1_Results/all_results.txt
```

### 5. Visualization

```
python visualisation.py --dataset-dir Task1_Datasets/parking_dataset_sample.txt --query-dir Task1_Datasets/query_points.txt --results-dir Task1_Results 
```
This will generate performance plots and visualize the R-Tree structure.

### 6. Compare Insertion Methods

```
python compare_insertion_methods.py --max-entries 10 --dataset-dir Task1_Datasets/parking_dataset_sample.txt --query-dir Task1_Datasets/query_points.txt --results-dir Task1_Results
```
This script builds the tree using both incremental insertion and bulk loading, then saves a figure with two bar charts (build and query times) in `Task1_Results/insertion_comparison.png`.


## Notes
- You can change dataset and query file paths using the `--dataset-dir` and `--query-dir` arguments.
- The `RTree` class also provides a `bulk_load` method that uses the
  Sort-Tile-Recursive technique to build the tree from a list of points,
  which is faster than inserting incrementally.
- For more details, see the code comments and function docstrings.

## Contact
For questions, please refer to the code comments
