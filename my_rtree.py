import sys
import math
import numpy as np
from tqdm import tqdm
class Point:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

    def distance_to(self, other):
        # Euclidean distance to another point
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __repr__(self):
        # For easy printing
        return f"Point(id={self.id}, x={self.x}, y={self.y})"
class MBR:
    def __init__(self):
        self.is_empty = True # Initialize as empty
        self.x1 = None
        self.y1 = None
        self.x2 = None
        self.y2 = None

    def expand_to_include(self, point_or_mbr):
        """
        Expand this MBR to include a new point or another MBR."""
        # Update MBR to include a new point or another MBR
        if isinstance(point_or_mbr, Point):
            if self.is_empty:
                # First real content: set all corners to this point
                self.x1 = self.x2 = point_or_mbr.x
                self.y1 = self.y2 = point_or_mbr.y
                self.is_empty = False
            else:
                self.x1 = min(self.x1, point_or_mbr.x)
                self.y1 = min(self.y1, point_or_mbr.y)
                self.x2 = max(self.x2, point_or_mbr.x)
                self.y2 = max(self.y2, point_or_mbr.y)
        elif isinstance(point_or_mbr, MBR):
            if point_or_mbr.is_empty:
                # If the other MBR is empty, no change
                return
            if self.is_empty:
                # Copy the other MBR exactly
                self.x1 = point_or_mbr.x1
                self.y1 = point_or_mbr.y1
                self.x2 = point_or_mbr.x2
                self.y2 = point_or_mbr.y2
                self.is_empty = False
            else:
                self.x1 = min(self.x1, point_or_mbr.x1)
                self.y1 = min(self.y1, point_or_mbr.y1)
                self.x2 = max(self.x2, point_or_mbr.x2)
                self.y2 = max(self.y2, point_or_mbr.y2)

    def distance_to_point(self, point):
        if self.is_empty:
        # An “empty” rectangle has no real bounds, so treat its distance
        # to any query as “infinite” (or a very large number).
            return float('inf')
        # Min distance from MBR to point (for NN search)
        dx = max(self.x1 - point.x, 0, point.x - self.x2)
        dy = max(self.y1 - point.y, 0, point.y - self.y2)
        return np.sqrt(dx * dx + dy * dy)

    def contains_point(self, point):
        # True if point is inside this MBR
        return self.x1 <= point.x <= self.x2 and self.y1 <= point.y <= self.y2

    def intersects(self, other):
        # Check if this MBR overlaps with another MBR
        return not (self.x2 < other.x1 or self.x1 > other.x2 or
                    self.y2 < other.y1 or self.y1 > other.y2)
    def perimeter(self):
        # Calculate the area of the MBR
        if self.is_empty:
            return 0
        return 2 * ((self.x2 - self.x1) + (self.y2 - self.y1))
    def perimeter_increase(self, point):
        # Calculate the increase in perimeter if this MBR is expanded to include a new point
         if self.is_empty:
            # If currently empty, the new perimeter after adding point is zero (degenerate).
            return 0
         else:
            # If not empty, calculate the new perimeter after including the point
            # Compute as before
            new_x1 = min(self.x1, point.x)
            new_y1 = min(self.y1, point.y)
            new_x2 = max(self.x2, point.x)
            new_y2 = max(self.y2, point.y)
            original_perimeter = self.perimeter()
            new_perimeter = 2 * ((new_x2 - new_x1) + (new_y2 - new_y1))
            return new_perimeter - original_perimeter
    def __repr__(self):
        return f"MBR(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2})"
class Node(object): 
    def __init__(self, max_entries, is_leaf=True):
        self.id = 0
        self.is_leaf = is_leaf #if the node is a leaf node, set is_leaf to True, otherwise set it to False  
        self.max_entries = max_entries #maximum number of data points in a node
        self.entries = [] #a list to store the entries of the node
        self.parent = None
        self.MBR = MBR() #create a MBR for the node

    def is_overflow(self):
        #check if the node is overflow
        return len(self.entries) > self.max_entries

    def is_root(self):
        #return True if the node is a root node, return False if the node is not a root node
        return self.parent is None
    def update_mbr(self):
        """Update the MBR of this node based on its data points or child nodes."""
        #update the MBR of the node
        self.MBR = MBR() #create a new MBR
        if self.is_leaf:
            for point in self.entries:
                self.MBR.expand_to_include(point)
        else:
            for child in self.entries:
                self.MBR.expand_to_include(child.MBR)
    def add_entry(self, entry):
        """
        Append a Point (if leaf) or a Node (if internal) into self.entries,
        then update MBR (and set parent if entry is a Node).
        """
        self.entries.append(entry)
        if not self.is_leaf:
            entry.parent = self
        self.update_mbr()
class RTree(object): #R tree class
    def __init__(self, max_entries):
        self.max_entries = max_entries #maximum number of data points in a node
        self.root = Node(max_entries=self.max_entries) #Create a root

    # --------------------------------------------------------------------------- #
    # Insertion functions
    # --------------------------------------------------------------------------- #
    def insert(self, node, point): # insert p(data point) to u (MBR)
        if node.is_leaf: 
            node.add_entry(point) #add the data point and update the corresponding MBR
            if node.is_overflow():
                self.handle_overflow(node) #handel overflow for leaf nodes
            node.update_mbr() #update the MBR of the node
        else:
            sub_node = self.choose_subtree(node, point) #choose a subtree to insert the data point to miminize the perimeter sum
            self.insert(sub_node, point) #keep continue to check the next layer recursively
            node.update_mbr() #update the MBR for inserting the data point

    # return the child whose MBR requires the minimum increase in perimeter to cover p

    def handle_overflow(self, node):
        node1, node2 = self.split(node) #u1 u2 are the two splits returned by the function "split"
        # if u is root, create a new root with s1 and s2 as its' children
        if node.is_root():
            new_root = Node(max_entries=node.max_entries, is_leaf=False) #create a new root
            new_root.add_entry(node1)
            new_root.add_entry(node2)
            self.root = new_root
            new_root.update_mbr() #update the MBR of the new root
        # if u is not root, delete u, and set s1 and s2 as u's parent's new children
        else:
            parent = node.parent #get the parent of the node
            # copy the information of s1 into u
            #parent.child_nodes.remove(node) #remove the node from the parent
            parent.entries = [c for c in parent.entries if c is not node]
            parent.add_entry(node1) #add the first child to the parent
            parent.add_entry(node2) #add the second child to the parent
            # update the MBR of the parent
            parent.update_mbr()
            if parent.is_overflow(): #check the parent node recursively
                self.handle_overflow(parent) #handle the overflow of the parent node


    def choose_subtree(self, node, point): 
        if node.is_leaf: #find the leaf and insert the data point
            return node
        else:
            min_increase = sys.maxsize #set an initial value
            best_child = None
            for child in node.entries: #check each child to find the best node to insert the point 
                increase = child.MBR.perimeter_increase(point)
                if increase < min_increase:
                    min_increase = increase
                    best_child = child
            return best_child

            
    def split(self, u):
        # split u into s1 and s2
        best_s1 = Node(max_entries=u.max_entries)
        best_s2 = Node(max_entries=u.max_entries)
        best_perimeter = sys.maxsize # set an initial value
        m = len(u.entries) # get the number of data points or child nodes
        minfill = math.ceil(0.4 * u.max_entries) # set the minimum fill factor to 0.4 of the maximum number of entries
        maxfill = m - minfill + 1 # set the maximum fill factor to m - minfill + 1
        # u is a leaf node
        if u.is_leaf:
            # create two different kinds of divides
            # divide the points based on X dimension and Y dimension
            # sort the points based on X dimension and Y dimension
            divides = [sorted(u.entries, key=lambda p: p.x),
                       sorted(u.entries, key=lambda p: p.y)]#sorting the points based on X dimension and Y dimension

        # u is a internal node
        else:
            # create four different kinds of divides
            # divide the points based on X1, X2, Y1, Y2
            # sort the points based on X1, X2, Y1, Y2
            divides = [sorted(u.entries, key=lambda child_node: child_node.MBR.x1), #sorting based on MBRs
                       sorted(u.entries, key=lambda child_node: child_node.MBR.x2),
                       sorted(u.entries, key=lambda child_node: child_node.MBR.y1),
                       sorted(u.entries, key=lambda child_node: child_node.MBR.y2)]
        # check the combinations to find a near-optimal one
        for divide in divides:
            for i in range(minfill, maxfill): #check the combinations
                # add the first half of the points to s1
                if u.is_leaf:
                    s1 = Node(max_entries=u.max_entries, is_leaf=True)
                    s2 = Node(max_entries=u.max_entries, is_leaf=True)
                else:
                    s1 = Node(max_entries=u.max_entries, is_leaf=False)
                    s2 = Node(max_entries=u.max_entries, is_leaf=False)
                s1.entries = divide[0: i]
                s1.update_mbr()
                # add the second half of the points to s2
                s2.entries = divide[i: len(divide)]
                s2.update_mbr()
                # check the perimeter
                # if the perimeter of s1 and s2 is smaller than the current minimum perimeter, update the minimum perimeter
                if best_perimeter > s1.MBR.perimeter() + s2.MBR.perimeter():
                    # update the best s1 and s2
                    best_perimeter = s1.MBR.perimeter() + s2.MBR.perimeter()
                    best_s1 = s1
                    best_s2 = s2
        # set the parent of s1 and s2 to u
        # set the parent of s1 to u
        for child in best_s1.entries:
            # set the parent of s1 to u
            child.parent = best_s1
        # set the parent of s2 to u
        for child in best_s2.entries:
            # set the parent of s2 to u
            child.parent = best_s2

        return best_s1, best_s2

        
    def nearest_neighbor(self, query_point, node=None, best_dist=float('inf'), best_point=None):
        """
        Nearest neighbor search using best-first search.
        Returns (best_dist, best_point_dict).
        """
        # If no node is provided, start from the root
        if node is None:
            node = self.root

        q = []
        q.append((node, node.MBR.distance_to_point(query_point))) #push the root node and its MBR distance to the queue
        q.sort(key=lambda x: x[1]) #sort the queue based on the MBR distance
        while q:
            current_node, mbr_dist = q.pop(0)
            # If the MBR distance is greater than the best distance, prune the search
            if mbr_dist > best_dist:
                break
            if current_node.is_leaf:
                # Check all points in the leaf node
                for data_point in current_node.entries:
                    #dist = self._point_distance((query_point['x'], query_point['y']), (data_point['x'], data_point['y']))
                    dist = query_point.distance_to(data_point) #calculate the distance between the query point and the data point
                    if dist < best_dist:
                        # Update the best distance and point
                        best_dist = dist
                        best_point = data_point
            else:
                # Prepare a list of (min_dist_to_MBR, child_node)
                for child in current_node.entries:
                    #mbr = (child.MBR['x1'], child.MBR['y1'], child.MBR['x2'], child.MBR['y2'])
                    #min_dist = self._mbr_min_dist((query_point['x'], query_point['y']), mbr)
                    min_dist = child.MBR.distance_to_point(query_point) #calculate the distance between the query point and the MBR of the child node
                    q.append((child, min_dist))
                # Sort the queue based on the MBR distance
                q.sort(key=lambda x: x[1]) #sort the queue based on the MBR distance
        return best_dist, best_point

    def fit(self, data_points, show_progress=True):
        '''Fit the R-tree with a list of data points.
        This method inserts all data points into the R-tree.
        '''
        iterable = tqdm(data_points, desc="Inserting") if show_progress else data_points
        for p in iterable:
            self.insert(self.root, p)
        return self
    
    def search(self, query_points, show_progress=True):
        """Search for nearest neighbors for a list of query points."""
        results = []
        iterable = tqdm(query_points, desc="Searching for nearest neighbors") if show_progress else query_points
        for q in iterable:
            best_dist, best_point = self.nearest_neighbor(q)
            results.append((best_point.id, best_point.x, best_point.y, q.id, best_dist))
        return results

    def bulk_load(self, data_points):
        """Build the tree using the Sort-Tile-Recursive (STR) algorithm."""
        if not data_points:
            return self

        def str_partition(entries, is_leaf):
            """Group entries into nodes using STR."""
            capacity = self.max_entries
            n = len(entries)
            num_nodes = math.ceil(n / capacity)
            slice_count = math.ceil(math.sqrt(num_nodes))
            slice_size = math.ceil(n / slice_count)

            key_x = (lambda e: e.x) if is_leaf else (lambda e: e.MBR.x1)
            key_y = (lambda e: e.y) if is_leaf else (lambda e: e.MBR.y1)

            entries = sorted(entries, key=key_x)
            nodes = []
            for i in range(0, n, slice_size):
                tile = entries[i:i + slice_size]
                tile = sorted(tile, key=key_y)
                for j in range(0, len(tile), capacity):
                    node = Node(max_entries=self.max_entries, is_leaf=is_leaf)
                    node.entries = tile[j:j + capacity]
                    if not is_leaf:
                        for child in node.entries:
                            child.parent = node
                    node.update_mbr()
                    nodes.append(node)
            return nodes

        nodes = str_partition(list(data_points), is_leaf=True)
        while len(nodes) > 1:
            nodes = str_partition(nodes, is_leaf=False)

        self.root = nodes[0]
        self.root.parent = None
        return self
    