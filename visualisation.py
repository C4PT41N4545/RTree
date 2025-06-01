import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from my_rtree import RTree
from utils import read_point
import time
import argparse
from pathlib import Path
from tqdm import tqdm
def plot_results(results, results_dir='Task1_Results'):
    # Plot the results
    max_entries = [result[0] for result in results]
    total_time = [result[1] for result in results]
    average_time = [result[2] for result in results]

    plt.plot(max_entries, total_time, label='Total Time')
    plt.plot(max_entries, average_time, label='Average Time')
    plt.xlabel('Max Entries')
    plt.ylabel('Time (seconds)')
    plt.title('R-Tree Performance with Varying Max Entries')
    plt.legend()
    plt.grid()
    xticks = np.arange(min(max_entries), max(max_entries)+1, 1)
    plt.xticks(xticks)
    #save the plot to a file
    plt.savefig(results_dir / "RTree_Performance.png")
    plt.show()

def plot_rtree(rtree, results_dir, minmax):
    #plot the MBRs
    fig, ax = plt.subplots(figsize=(8,8))
    plot_node(rtree.root, ax)
    ax.set_xlim(minmax[0], minmax[2])
    ax.set_ylim(minmax[1], minmax[3])
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('R-Tree MBR Visualization')
    ax.grid(True)
    plt.savefig(results_dir / "RTree_Structure.png")
    plt.show()

def plot_node(node, ax, depth=0):
    # pick a colour per depth if you like:
    colours = ['blue','red','orange','purple']
    edge = colours[depth % len(colours)]
    # draw this node’s MBR
    x1,y1 = node.MBR.x1, node.MBR.y1
    w,h = node.MBR.x2-x1, node.MBR.y2-y1
    ax.add_patch(Rectangle((x1,y1), w, h, 
                           fill=False, edgecolor=edge, linewidth=2))
    #print depth
    #print("depth=", depth, "MBR=", node.MBR)
    if node.is_leaf:
        # plot all its points
        xs = [p.x for p in node.data_points]
        ys = [p.y for p in node.data_points]
        #ax.scatter(xs, ys, c='green', zorder=5)
        #print("data points=", node.data_points)

    else:
        # recurse
        for child in node.child_nodes:
            plot_node(child, ax, depth+1)
def find_best_max_entries(data_points, query_points):
    # Example: find the best max_entries for the R-Tree
    # This is a placeholder function. You can implement your own logic to find the best max_entries.
    # For example, you can use cross-validation or other techniques to find the best max_entries.
    results = []
    for i in tqdm(range(4, 15+1), desc="Finding best max_entries", unit="max_entries"):
        # Create an R-Tree with the current max_entries
        rtree = RTree(max_entries=i)
        # Insert the data points into the R-Tree
        rtree.fit(data_points, show_progress=False)
        # Measure the time taken for nearest neighbor search
        total_start_time = time.time()
        for query_point in query_points:
            # Perform nearest neighbor search
            best_dist, best_point = rtree.nearest_neighbor(query_point)
        total_end_time = time.time()
        # Calculate the average time for the current max_entries
        total_time = total_end_time - total_start_time
        average_time = total_time / len(query_points)
        results.append((i,total_time, average_time))
    return results  # Return a default value for now

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run nearest neighbor search algorithms.")
    parser.add_argument('--dataset-dir', type=Path, default=Path('Task1_Datasets/parking_dataset.txt'), help='Directory containing the dataset.')
    parser.add_argument('--query-dir', type=Path, default=Path('Task1_Datasets/query_points.txt'), help='Directory containing the dataset.')
    parser.add_argument('--results-dir', type=Path, default=Path('Task1_Results'), help='Directory to save the results.')
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    query_dir = args.query_dir
    results_dir = args.results_dir
    # Read data points from file
    data_points = read_point(dataset_dir)
    query_points = read_point(query_dir)
    # Find the best max_entries for the R-Tree
    results = find_best_max_entries(data_points, query_points)
    #find min and max x an y in data_points
    min_x = min(p.x for p in data_points)
    min_y = min(p.y for p in data_points)
    max_x = max(p.x for p in data_points)
    max_y = max(p.y for p in data_points)
    minmax = (min_x, min_y, max_x, max_y)
    # Plot the results
    plot_results(results, results_dir)
    # Create an R-Tree with the best max_entries
    best_max_entries = results[-1][0]
    #best_max_entries=10
    rtree = RTree(max_entries=best_max_entries)
    rtree.fit(data_points)
    
    # Plot the R-Tree structure
    plot_rtree(rtree, results_dir, minmax)
if __name__ == "__main__":
    main()