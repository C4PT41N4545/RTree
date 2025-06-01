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
        self.child_nodes = [] #for non-leaf node, a list to store nodes
        self.data_points = [] #for leaf node, a list to store data points
        self.parent = None
        self.MBR = MBR() #create a MBR for the node

    def is_overflow(self):
        #check if the node is overflow
        #if the number of data points is greater than B, return True
        if self.is_leaf:
            return len(self.data_points) > self.max_entries
        else:
            return len(self.child_nodes) > self.max_entries

    def is_root(self):
        #return True if the node is a root node, return False if the node is not a root node
        return self.parent is None
    def update_mbr(self):
        """Update the MBR of this node based on its data points or child nodes."""
        #update the MBR of the node
        if self.is_leaf:
            self.MBR = MBR() #create a new MBR
            for point in self.data_points:
                self.MBR.expand_to_include(point)
        else:
            self.MBR = MBR() #create a new MBR
            for child in self.child_nodes:
                self.MBR.expand_to_include(child.MBR)
class RTree(object): #R tree class
    def __init__(self, max_entries):
        self.max_entries = max_entries #maximum number of data points in a node
        self.root = Node(max_entries=self.max_entries) #Create a root

    # --------------------------------------------------------------------------- #
    # Insertion functions
    # --------------------------------------------------------------------------- #
    def insert(self, node, point): # insert p(data point) to u (MBR)
        if node.is_leaf: 
            self.add_data_point(node, point) #add the data point and update the corresponding MBR
            if node.is_overflow():
                self.handle_overflow(node) #handel overflow for leaf nodes
            node.update_mbr() #update the MBR of the node
        else:
            sub_node = self.choose_subtree(node, point) #choose a subtree to insert the data point to miminize the perimeter sum
            self.insert(sub_node, point) #keep continue to check the next layer recursively
            node.update_mbr() #update the MBR for inserting the data point

    def add_data_point(self, node, data_point): #add data points and update the the MBRS
        # 1. add the point into node.data_points
        node.data_points.append(data_point)
        # 2. update node.MBR\
        node.update_mbr() #update the MBR of the node
        

    def add_child(self, node, child):
        # 1. add the child into node.child_nodes
        node.child_nodes.append(child) #add the child into the node
        # 2. set the node as the parent of the child
        child.parent = node #set the parent of the child
        # 3. update node.MBR
        node.update_mbr() #update the MBR of the node

    # return the child whose MBR requires the minimum increase in perimeter to cover p

    def handle_overflow(self, node):
        node1, node2 = self.split(node) #u1 u2 are the two splits returned by the function "split"
        # if u is root, create a new root with s1 and s2 as its' children
        if node.is_root():
            new_root = Node(max_entries=node.max_entries, is_leaf=False) #create a new root
            self.add_child(new_root, node1)
            self.add_child(new_root, node2)
            self.root = new_root
            new_root.update_mbr() #update the MBR of the new root
        # if u is not root, delete u, and set s1 and s2 as u's parent's new children
        else:
            parent = node.parent #get the parent of the node
            # copy the information of s1 into u
            #parent.child_nodes.remove(node) #remove the node from the parent
            parent.child_nodes = [c for c in parent.child_nodes if c is not node]
            self.add_child(parent, node1) #add the first child to the parent
            self.add_child(parent, node2) #add the second child to the parent
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
            for child in node.child_nodes: #check each child to find the best node to insert the point 
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
        # u is a leaf node
        if u.is_leaf:
            m = u.data_points.__len__() # get the number of data points
            # create two different kinds of divides
            # divide the points based on X dimension and Y dimension
            # sort the points based on X dimension and Y dimension
            divides = [sorted(u.data_points, key=lambda p: p.x),
                       sorted(u.data_points, key=lambda p: p.y)]#sorting the points based on X dimension and Y dimension
            for divide in divides:
                # check the combinations to find a near-optimal one
                for i in range(math.ceil(0.4 * u.max_entries), m - math.ceil(0.4 * u.max_entries) + 1): #check the combinations to find a near-optimal one
                    # add the first half of the points to s1
                    s1 = Node(max_entries=u.max_entries)
                    s1.data_points = divide[0: i] #add the first half of the points to s1
                    s1.update_mbr()
                    s2 = Node(max_entries=u.max_entries)
                    s2.data_points = divide[i: divide.__len__()] #add the second half of the points to s2
                    s2.update_mbr()
                    if best_perimeter > s1.MBR.perimeter() + s2.MBR.perimeter(): #check the perimeter
                        # if the perimeter of s1 and s2 is smaller than the current minimum perimeter, update the minimum perimeter
                        best_perimeter = s1.MBR.perimeter() + s2.MBR.perimeter()
                        # update the best s1 and s2
                        best_s1 = s1
                        best_s2 = s2

        # u is a internal node
        else:
            # create four different kinds of divides
            # divide the points based on X1, X2, Y1, Y2
            # sort the points based on X1, X2, Y1, Y2
            m = u.child_nodes.__len__() #get the number of child nodes
            divides = [sorted(u.child_nodes, key=lambda child_node: child_node.MBR.x1), #sorting based on MBRs
                       sorted(u.child_nodes, key=lambda child_node: child_node.MBR.x2),
                       sorted(u.child_nodes, key=lambda child_node: child_node.MBR.y1),
                       sorted(u.child_nodes, key=lambda child_node: child_node.MBR.y2)]
            # check the combinations to find a near-optimal one
            for divide in divides:
                for i in range(math.ceil(0.4 * u.max_entries), m - math.ceil(0.4 * u.max_entries) + 1): #check the combinations
                    # add the first half of the points to s1
                    s1 = Node(max_entries=u.max_entries, is_leaf=False)
                    s1.child_nodes = divide[0: i]
                    s1.update_mbr()
                    # add the second half of the points to s2
                    s2 = Node(max_entries=u.max_entries, is_leaf=False)
                    s2.child_nodes = divide[i: divide.__len__()]
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
        for child in best_s1.child_nodes:
            # set the parent of s1 to u
            child.parent = best_s1
        # set the parent of s2 to u
        for child in best_s2.child_nodes:
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
                for data_point in current_node.data_points:
                    #dist = self._point_distance((query_point['x'], query_point['y']), (data_point['x'], data_point['y']))
                    dist = query_point.distance_to(data_point) #calculate the distance between the query point and the data point
                    if dist < best_dist:
                        # Update the best distance and point
                        best_dist = dist
                        best_point = data_point
            else:
                # Prepare a list of (min_dist_to_MBR, child_node)
                for child in current_node.child_nodes:
                    #mbr = (child.MBR['x1'], child.MBR['y1'], child.MBR['x2'], child.MBR['y2'])
                    #min_dist = self._mbr_min_dist((query_point['x'], query_point['y']), mbr)
                    min_dist = child.MBR.distance_to_point(query_point) #calculate the distance between the query point and the MBR of the child node
                    q.append((child, min_dist))
                # Sort the queue based on the MBR distance
                q.sort(key=lambda x: x[1]) #sort the queue based on the MBR distance
        return best_dist, best_point

    def fit(self, data_points):
        '''Fit the R-tree with a list of data points.
        This method inserts all data points into the R-tree.
        '''
        print("Fitting R-tree with data points...")
        for p in tqdm(data_points, desc="Inserting"):
            self.insert(self.root, p)
        return self

    