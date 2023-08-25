"""FIT2004 Assignment 1 File"""
__author__ = "Brandon Yong Hoong Tak (32025963)"

###########################################################################################################################################
# MIN HEAP                                                                                                                                #
###########################################################################################################################################
"""
    Class description: Minimum heap data structure primarily used as a priority queue in dijkstras algorithm.
    Based of the maximum heap implementation provided in FIT1008, has been modified slightly to work with 0 indexing, hence children of index k will be located
    at indexes 2k+1 and 2k+2 respectively.
"""
class MinHeap:

    MIN_CAPACITY = 1

    def __init__(self, capacity):
        """
        Function description: Constructor for the heap object. Initializes two arrays of size capacity, the first is used as the actual heap array
        where elements are inserted and removed while the second array acts as a mapping of objects to their index position within the first array.
        This allows us to easily find the location of integer elements within our heap by looking up their index within the second array.

        Allocating the space for both array is O(n) + O(n) = O(n) for both time and space complexity

        Input:
            capacity: An integer value for the maximum capacity of the heap

        Output: None

        Time complexity: O(n) where n is the capacity of the heap
        Aux space complexity: O(n) where n is the capacity of the heap
        """
        self.arr = [None]*max(self.MIN_CAPACITY, capacity)
        self.map = [None]*max(self.MIN_CAPACITY, capacity)  # index position = element id, value = index position within the heap
        self.length = 0

    def __len__(self):
        """
        Function description: Returns the number of items in the heap

        Input: None

        Output: The number of elements in the heap

        Time complexity: O(1)
        Aux space complexity: O(1)
        """
        return self.length
    
    def smallest_child(self, k):
        """
        Function description: Returns the index of the smallest child of index k within the heap

        Input:
            k: The index position to check at

        Output: The index position of the smallest child of k

        Time complexity: O(1)
        Aux space complexity: O(1)
        """
        if 2*k + 1 == self.length or self.arr[2*k + 1] < self.arr[2*k + 2]: # if there is no second child, or the first child is smaller
            return 2*k + 1
        else:                                                               # the second child is smaller
            return 2*k + 2
    
    def rise(self, k):
        """
        Function description: Rises an element in the heap at index position k to its correct position within the heap by swapping elements on the way up.
        Maintains the heap property when used in conjunction with decrease key and add operations.

        Since a heap is a complete binary tree, in the worst case we must perform O(log n) swaps to rise our element to the top of the heap/root.

        Input:
            k: The index of the element to begin rising

        Output: None

        Time complexity: O(log n), where n is the number of elements in the heap
        Aux space complexity: O(1)
        """
        item = self.arr[k]
        # while k is not at the root, and the element at k is smaller than its parent
        while k > 0 and item < self.arr[(k - 1)//2]:
            self.arr[k] = self.arr[(k - 1)//2]          # swap the parent and child
            self.map[int(self.arr[(k - 1)//2])] = k     # update the mapping of the swapped element
            k = (k - 1)//2                              # update index k to its parent index
        self.arr[k] = item
        self.map[int(item)] = k                         # ensure the mapping is correct

    def sink(self, k):
        """
        Function description: Sinks an element in the heap at index position k to its correct position within the heap by swapping elements on the way down.
        Maintains the heap property when used in conjunction with decrease key and add operations.

        Since a heap is a complete binary tree, in the worst case we must perform O(log n) swaps to sink our element to the bottom of our heap

        Input:
            k: The index of the element to begin sinking

        Output: None

        Time complexity: O(log n), where n is the number of elements in the heap
        Aux space complexity: O(1)
        """
        item = self.arr[k]
        # while we are not at the bottom of the heap
        while 2*k + 1 <= self.length:
            min_child = self.smallest_child(k)          # find the smaller element between k's children
            if self.arr[min_child] >= item:             # if k's smallest child is bigger or equal to k's element, we are in the correct position
                break
            else:
                self.arr[k] = self.arr[min_child]       # swap parent and child
                self.map[int(self.arr[min_child])] = k  # update the mapping of the swapped element
                k = min_child
        self.arr[k] = item
        self.map[int(item)] = k                         # ensure the mapping is correct

    def add(self, item):
        """
        Function description: Adds an item to the bottom of the heap and then rise it to its correct position within the heap.

        In the worst case, our rise operation may perform O(log n) swaps, we have a complexity of O(log n)

        Input:
            item: The item to be added

        Output: None

        Time complexity: O(log n), where n is the number of elements in the heap
        Aux space complexity: O(1)
        """
        if self.length == len(self.arr):
            raise IndexError
        
        self.arr[self.length] = item    # insert our item at the bottom of the heap
        self.rise(self.length)          # rise it, O(log n)
        self.length += 1

    def extract_min(self):
        """
        Function description: Removes and returns the smallest item in our priority queue.
        Before extracting our minimum value from the root node, we first swap it with the last element in our heap and then sink the element from the root position
        to its correct position within the heap.

        Finding the smallest item is trivial since it is located at index 0, however after swapping the root and bottom element, we may perform
        O(log n) swaps within our sink operation.

        Input: None

        Output: The smallest element in our heap

        Time complexity: O(log n), where n is the number of elements in the heap
        Aux space complexity: O(1)
        """
        if self.length == 0:
            raise IndexError

        min_item = self.arr[0]                      # save the smallest element
        self.length -= 1
        if self.length > 0:
            self.arr[0] = self.arr[self.length]     # swap the root element with the bottom element
            self.sink(0)                            # sink it from the root to its correct position
        return min_item
    
    def decrease_key(self, key, amount):
        """
        Function description: Decreases the value of an element within our heap.
        By doing so we can increase its priority within the queue which will result in the element being served faster.
        Makes use of the mapping we created to find the index position of specific elements within the heap in O(1) time. 
        This mapping is usually implemented as a hashmap but due to assignment constraints an array is used where the array's index position acts as a numerical key 
        and the value at said index position is the index position of the element within the heap.

        Since finding the element O(1) and the cost of rising it to its new position is O(log n), the overall complexity
        of this operation is O(log n)

        Input:
            key: The numerical key used to find the index position of our element within the heap.
            amount: The amount we wish to decrease our element's key by

        Output: None

        Time complexity: O(log n), where n is the number of elements in the heap
        Aux space complexity: O(1)
        """
        index = self.map[key]       # find the element
        self.arr[index] - amount    # decrease its key
        self.rise(index)            # rise it to its new position

###########################################################################################################################################
# GRAPH STUFF                                                                                                                             #
###########################################################################################################################################
"""
    Class description: Class representing a graph object using an adjacency list.
    Based of the graph implementation shown in the FIT2004 lecture videos provided by Dr Lim Wern Han
"""
class Graph:

    def __init__(self, V):
        """
        Function description: Constructor for the graph object.
        Initializes an array of vertices of size V and creates a vertex object at each index

        Creating an array of size V is O(V) time and space comp

        Input:
            V: The number of vertices in the graph

        Output: None

        Time complexity: O(V) where V is the number of vertices in the graph
        Aux space complexity: O(V) where V is the number of vertices in the graph
        """
        # create an empty array of size V
        self.vertices = [None]*V
        # now assign each vertex to its position within the array
        for i in range(V):
            self.vertices[i] = Vertex(i)
    
    def dijkstras_algorithm(self, start):
        """
        Function description: Standard implementation of Dijkstras shortest path algorithm using a minimum heap. 
        Finds the shortest path from the source/start node to every other node in the graph.
        Based of the graph implementation shown in the FIT2004 lecture videos provided by Dr Lim Wern Han 

        Our time complexity comes from the fact that in the worst case we must relax each edge exactly once (decrease key), resulting in a complexity of O(E log V) 
        since we also have to extract each vertex from our heap at least once we also have O(V log V), resulting in a combined cost of O(E log V + V log V)
        which reduces to O(E log V)

        Our aux space complexity is the result of creating a minimum heap of size V as the discovered queue, resulting in O(V)

        Input:
            start: The source/starting vertex

        Output: None

        Time complexity: O(E log V) where E is the number of edges in the graph and V is the number of vertices
        Aux space complexity: O(V) where V is the number of vertices in the graph
        """
        # create the priority queue and add the source node
        discovered = MinHeap(len(self.vertices))
        start_vertex = self.vertices[start]
        start_vertex.distance = 0
        discovered.add(start_vertex)

        # while we havent discovered and visited each node
        while len(discovered) > 0:

            # get the current node with the shortest distance from the source and visit it
            curr_location = discovered.extract_min()
            curr_location.visited = True

            # for each of its edges
            for edge in curr_location.edges:
                neighbour = edge.end    # get the edges destination node                

                # if it hasnt been discovered before
                if neighbour.discovered == False: 
                    neighbour.discovered = True
                    neighbour.distance = curr_location.distance + edge.weight
                    neighbour.previous = curr_location
                    discovered.add(neighbour)   # add it to the priority queue    

                # if it has been discovered, but not visited, check if we have found a shorter distance
                elif neighbour.visited == False:

                    # if the neighbours currently known distance from source is greater than the newly discovered distance, perform edge relaxation
                    if neighbour.distance > curr_location.distance + edge.weight:
                        difference = neighbour.distance - (curr_location.distance + edge.weight)
                        neighbour.distance = curr_location.distance + edge.weight
                        neighbour.previous = curr_location

                        # now update the priority queue
                        discovered.decrease_key(neighbour.id, difference)

"""
    Class description: Class representing a vertex object.
    Some of the comparison magic methods for this class have been over written allowing vertices to be compared based on distance when used in a heap.
    The int() and subtraction magic methods have also been over written so converting a vertex to an integer returns its id 
    and subtraction operations directly subtract from the distance attribute.
    Based of the graph implementation shown in the FIT2004 lecture videos provided by Dr Lim Wern Han
"""
class Vertex:

    def __init__(self, id):
        """
        Function description: Constructor for the vertex object.
        Initializes an empty list of edges representing the vertices neighbouring this one.
        Initializes some attributes used in dijkstras algorithm

        Input:
            id: The numerical id of the vertex

        Output: None

        Time complexity: O(1)
        Aux space complexity: O(1)
        """
        self.id = id
        self.edges = []

        # djikstra related
        self.distance = float('inf')
        self.discovered = False
        self.visited = False
        self.previous = None
    
    def __int__(self):
        """
        Function description: Returns the id of the vertex

        Input: None

        Output: The id of the vertex as an int

        Time complexity: O(1)
        Aux space complexity: O(1)
        """
        return self.id
    
    def __sub__(self, other):
        """
        Function description: Subtracts some integer amount from this vertexes distance attribute

        Input:
            Other: Integer amount to subtract from distance

        Output: None

        Time complexity: O(1)
        Aux space complexity: O(1)
        """
        return self.distance - other
    
    def __lt__(self, other):
        """
        Function description: Checks if this.distance < other.distance

        Input: 
            Other: The vertex to check against

        Output: True or False

        Time complexity: O(1)
        Aux space complexity: O(1)
        """
        return self.distance < other.distance

    def __le__(self, other):
        """
        Function description: Checks if this.distance <= other.distance

        Input: 
            Other: The vertex to check against

        Output: True or False

        Time complexity: O(1)
        Aux space complexity: O(1)
        """
        return self.distance <= other.distance

    def __gt__(self, other):
        """
        Function description: Checks if this.distance > other.distance

        Input: 
            Other: The vertex to check against

        Output: True or False

        Time complexity: O(1)
        Aux space complexity: O(1)
        """
        return self.distance > other.distance

    def __ge__(self, other):
        """
        Function description: Checks if this.distance >= other.distance

        Input:
            Other: The vertex to check against

        Output: True or False

        Time complexity: O(1)
        Aux space complexity: O(1)
        """
        return self.distance >= other.distance

"""
    Class description: Class representing an edge object with a start, end and weigth value.
    Based of the graph implementation shown in the FIT2004 lecture videos provided by Dr Lim Wern Han
"""
class Edge:

    def __init__(self, start, end, weight):
        """
        Function description: Constructor for the edge object.
        Sets the objects attributes to their appropriate values.

        Input:
            start: The source vertex object (from)
            end: The destination vertex object (to)
            weight: The edge weight

        Output: None

        Time complexity: O(1)
        Aux space complexity: O(1)
        """
        self.start = start
        self.end = end
        self.weight = weight
    
###########################################################################################################################################
# ASSIGNMENT FUNCTION 1                                                                                                                   #
###########################################################################################################################################

def optimalRoute(start, end, passengers, roads):
    """
        1) Should I give a ride? (10 marks)

        Function description: Given a start, end, list of passenger locations and a list of edges with their associated weights, this function
        computes the shortest distance from the start to the end using a standard implementation of dijkstras's algorithm, and returns the route in the form of a list of integers.

        Approach description: To optimise our drive time, we have to essentially make a decision if it is more efficient to path towards the end node directly without passing through a passenger node or if it is more efficient to pick up a passenger in order to access the faster carpool lanes. 
        This can be determined by modelling the problem as 2 graphs with identical node and edge configurations but different weights. 
        The first graph which we call the "real" graph will have the standard set of edge weights with nodes 0...L-1, while the second graph called the "carpool" graph will have the carpool set of weights with nodes L...2*L-1, 
        we can then connect these two graphs at their corresponding passenger nodes using 0 weight edges resulting in one unified graph where traversing to a passenger node allows us to access the carpool section of the graph with better edge weights. 
        In this way, we can think of the problem as having two "end" destinations, the first of which being the end destination located in the "real" graph and the second being in the "carpool" graph, hence if our real end destination is at node N, our carpool end destination will be at node N+L. 
        This makes our problem much simpler as now our problem can simply be interpreted as "Is it faster to go to the "real" end node or the "carpool" end node?", which we can solve by running Dijkstra's algorithm on our graph to determine the shortest path from our start node to both the real end and carpool end destinations. 
        We can then compare the distance from both end destinations to determine if the shortest path involved passing through a passenger node or not, after that we can simply backtrack from the end node with the shorted distance to obtain the optimal route.

        The overall time complexity of our algorithm can be found by analyzing the time complexity of the 3 major steps of our algorithm.
        The first step is to construct our graph from the list of edges, this  involves creating an array of size 2L to store our vertices,
        as well as looping over the list of edges twice and our passenger list once (P = O(L)), resulting in a complexity of O(3L + 2R) which is O(L + R). 
        The second step involves running dijkstra's algorithm on our graph which results in a time complexity of O(R log L).
        Finally, backtracking from the end node is at most O(2L) which is O(L). Putting it all together, we arrive at O(L + R) + O(R log L) + O(L),
        which reduces to O(R log L) the dominant term.

        This algorithm uses auxilliary space in 3 different places, the first of which is the adjacency list we use to store our graph which costs O(L + R),
        the priority queue used by dijkstra's algorithm which is O(L) and the output list containing the optimal route which is also O(L),
        resulting in a final auxilliary space complexity of O(L + R).

        Input:
            start: The starting location as an integer
            end: The destination location as an integer
            passengers: A list of integers representing locations that contain passengers
            roads: A list of tuples representing the roads connecting each location.
                   each tuple contains 4 integers which are the start location, end location, standard weight and carpool weight respectively

        Output: The shortest route from start to end in the form of a list of integers.

        Time complexity: O(R log L) where R is the number of roads and L is the number of locations
        Aux space complexity: O(L + R) where R is the number of roads and L is the number of locations
    """
    
    # step 1, find the maximum node id from the list of roads/edges
    maximum = 0
    for road in roads:  # O(R) time comp
        maximum = max(maximum, road[0], road[1]) # we compare 3 items at each iteration so it is O(1)
    total_locations = maximum + 1                # now we know the total number of nodes is maximum + 1 (because we are zero indexed)

    # step 2, create a graph with with double the total number of locations
    # this allows us to use a layered graph approach where the first set of nodes have edges with standard weights while the second has carpool weights
    road_graph = Graph(2*total_locations) # O(2L) time comp and O(2L) space comp -> O(L) for both

    # step 3, connect the nodes from each graph with the correct edges
    # by modelling the problem like this, we have essentially created two graphs with different weights that are not connected to each other
    # i will refer to these two graphs as the standard graph and carpool graph
    for road in roads:  # O(R) time comp

        # unpack the values for clarity
        start_vertex = road_graph.vertices[road[0]]
        end_vertex = road_graph.vertices[road[1]]
        standard_weight = road[2]

        # we can use total locations as an offset to access the identical node from the carpool graph
        # for example, location 2 in the standard graph would correspond with location 7 in the carpool graph in this case
        carpool_start_vertex = road_graph.vertices[road[0] + total_locations]
        carpool_end_vertex = road_graph.vertices[road[1] + total_locations]
        carpool_weight = road[3]

        # now create the edge for start to end for both the standard graph and the carpool graph
        # this means we create 2R edges in our adjacency list in total, resulting in an aux space of O(R)
        start_vertex.edges.append(Edge(start_vertex, end_vertex, standard_weight))
        carpool_start_vertex.edges.append(Edge(carpool_start_vertex, carpool_end_vertex, carpool_weight))

    # step 4, we can now connect the standard and carpool graphs to each other by creating a 0 weighted edge from each node passenger node in the standard graph to its equivalent node in the carpool graph
    # Since we create an additional P edges and P = O(L - 2), this represents an additional O(L) time and aux space complexity
    for passenger in passengers:    # O(L) time comp   
        vertex = road_graph.vertices[passenger]
        carpool_vertex = road_graph.vertices[passenger + total_locations]
        vertex.edges.append(Edge(vertex, carpool_vertex, 0))

    # This brings our time complexity up to this point to O(R) + O(L) + O(R) + O(L) -> O(R + L)
    # and our total aux space complexity to O(L) + O(R) + O(L) -> O(L + R), this the worst case auxilliary space needed to store our graph as an adjacency list
        
    # step 5, now we can call djikstras algorithm to find the shortest path from start to end
    # this calls djikstra on a graph of size O(2L) vertices and O(2R + L).
    # This results in a time complexity of O((2R + L) log 2L), which simplifies to O((2R log 2L) + (L log 2L))
    # which is nothing but O(R log L) time complexity wise as the graph is connected and R >= L
    # we also use an auxilliary space of O(L) for the discovered queue here
    road_graph.dijkstras_algorithm(start)

    # step 6, back track from the end to find the shortest ditance to the start
    # first lets see if it is faster to travel to the carpool or standard end node
    end_vertex = min(road_graph.vertices[end], road_graph.vertices[end + total_locations]) # thanks to our overwritten comparison methods, we can use min on vertex objects to compare distances directly

    # now lets backtrack from the end node appending everything along the way
    # the list we create here can grow as large as O(2L) nodes and we may loop a total of O(2L) times resulting in a time and aux space comp of O(L)
    optimal_route = []
    while end_vertex.id != start:

        if (end_vertex.id - total_locations) == end_vertex.previous.id: # don't append traversal from standard passenger node to carpool passenger node, they are technically the same place
            pass
        elif end_vertex.id >= total_locations:                          # if the node is from the carpool graph, decrement its id by total_locations
            optimal_route.append(end_vertex.id - total_locations)
        else:                                                           
            optimal_route.append(end_vertex.id)                 
            
        end_vertex = end_vertex.previous

    optimal_route.append(end_vertex.id)         # finally we append the starting vertex and reverse the list
    optimal_route.reverse()                     # list reversal is O(L)

    return optimal_route                  

    # The final time complexity for our function is O(R + L) graph creation + O(R log L) dijkstras + O(L) backtracking + O(L) reversal
    # resulting in a time complexity of O(R log L) as it is the dominant term

    # The final aux space complexity for our function is O(L + R) adjacency list + O(L) dijkstras + O(L) output list
    # resulting in a auxilliary space complexity of O(L + R)

###########################################################################################################################################
# ASSIGNMENT FUNCTION 2                                                                                                                   #
###########################################################################################################################################

def select_sections(occupancy_probability):
    """
        2) Repurposing Underused Workspace (Dynamic Programming) (10 marks)

        Function description: Given a matrix of n rows and m columns of integers representing occupancy probability ranging from 0 to 100, this function computes
        the minimum total occupancy that can be achieved by traversing upwards directly or diagonally from row 0 to row n and returns a list
        of both the minimum total occupancy achievable as well as a list of tuples (n,m) representing the optimal column to travel to at each row from the previous.

        Approach description: This problem is essentially a dp maze problem where we are trying minimize the total distance/cost of traversing from start to end.
        In our case, we want to travel from row 0 to row n while taking the most optimal column m in order to minimize our total occupancy probability. 
        Hence at every row, we need to make a decision of which to take that will result in the lowest total occupancy in the end. Since we cant see into the future, at each stage to know
        which column will result in the best total occupancy is difficult, so instead we start from the end point which is row n. At row n the decision of which column to take is simple, as it is simply
        the column with the minimum occupancy of the row. However, if we go down a row, to row n - 1, we now need to make a decision again on which column minimizes the total occupancy on the way to the destination.
        But now we can use the knowledge of the previous row to help us, since we know that each row can only travel to one of the 3 or 2 (edge columns) columns above it, we know that the minimum total 
        occupancy cost of any column on a given row is simply the minimum total occupancy of the 3 columns above it plus its own occupancy cost, this applies to every row all the way down to
        n = 0 allowing us to find the best column to start at by taking the minimum total occupancy of the columns along the row. From that point onwards we simply need to backtrack from the optimal starting
        column all the way to the nth row, this will give us the optimal route since every decision that was made along the way to determine the total occupancy at every column was an optimal solution.

        Hence the solution can be described as follows:

        step 1 create an n*m size memo to store the minimum total occupancy of every position along a row. O(nm) time and aux space complexity

        step 2 iterate over every row starting from row n. On the first iteration, we just set the total occupancy of each column along the row to its own occupancy probability, this is our base case. 
        Then for each row all the way down to the start, we calculate and memoize the minimum total occupancy of each column along the row using the minimum of the previously calculated total occupancies from the 3 positions above it.
        There are a few edge cases, for columns on the far left or right, we can only take the minimum of two positions, the one directly above it and the one to its diagonal right or left. There is also the case of there being only
        one column, in that scenario we can just take the position directly above it without evaluating anything. O(nm) time complexity

        step 3 backtrack from the minimum starting position. We start at the optimal column along the nth row (bottom), which is the column with the minimum total occupancy.
        From there, finding the optimal column along the next row is straight forward, since our memo is populated with the minimum achievable total occupancies for each column along every row,
        all we have to do is check the 3 or 2 columns above our current optimal column on our current row in the memo, and travel to the column with the minimum total occupancy. Since all we
        are doing is some comparison operations and arithmetic while traversing to upwards from row n to 0, our back tracking is O(n) time complexity  and O(n) aux space for the output list.

        This solution is heavily inspired by the dp maze solutions discussed in the FIT2004 dp lecture videos provided by Dr Lim Wern Han 

        Input:
            occupancy_probability: A matrix with n rows and m columns, the elements of the matrix are integers 
                                   ranging from 0 to 100 representing the occupnacy  probability of that position

        Output: A list containing two elements, the first is the minimum total occupancy that can be achieved from the input matrix
                and the second is a list of tuples that describes the optimal route from the first row to the nth row in the form of (n,m)

        Time complexity: O(nm) where n is the number of rows and m is the number of columns in the matrix
        Aux space complexity: O(nm) where n is the number of rows and m is the number of columns in the matrix
    """

    n = len(occupancy_probability)
    m = len(occupancy_probability[0])
    
    # step 1, create our memo
    # allocating a space for the 2d memo matrix costs O(nm) aux space and time comp
    memo = [None]*n
    for i in range(n):
        memo[i] = [None]*m

    # step 2, iterate over all sections and commit the minimum total occupancy possibility that can be achieved by taking that path, to its corresponding memo position
    # O(nm) time complexity for the nested for loop
    for i in range(n):
        for j in range(m):
            if i == 0:          # base case, we reached the top row so the occupancy probability is just itself
                memo[i][j] = occupancy_probability[i][j]
            elif m == 1:        # if we have only one column, the only option is to go up (edge case)
                memo[i][j] = memo[i - 1][j] + occupancy_probability[i][j]
            elif j == 0:        # if we are at the first column, we can't go diagonally left, so take the minimum path between up and diagonally right
                memo[i][j] = min(memo[i - 1][j], memo[i - 1][j + 1]) + occupancy_probability[i][j]
            elif j == m - 1:    # if we are at the last column, we can't go diagonally right, so take the minimum of the path between up and diagonally left
                memo[i][j] = min(memo[i - 1][j], memo[i - 1][j - 1]) + occupancy_probability[i][j]
            else:               # if we are anywhere else, we take the minimum of the three possible routes upwards, this provides us with the optimal path to reach the node above us, which is the subproblem for the optimal path to the node above that
                memo[i][j] = min(memo[i - 1][j], memo[i - 1][j - 1], memo[i - 1][j + 1]) + occupancy_probability[i][j]

    # Step 3, backtracking to reconstruct our solution
    # first, we find the starting position with the minimum total occupancy
    # the min and index call here are O(m) time comp each
    minimum_total_occupancy = min(memo[n - 1])
    optimal_column_index = memo[n - 1].index(minimum_total_occupancy) 

    # initialize an empty list to append to
    sections_location = [] # can grow to O(n) aux space complexity

    for i in reversed(range(n)): # O(n) time comp, reversed is O(1) time comp
        
        sections_location.append((i, optimal_column_index)) # append the current row we are at along with the optimal column of the row

        if i != 0 and m != 1:    # if we are not at the first row and we have more than 1 column

            if optimal_column_index == 0: # far left
                up = memo[i - 1][optimal_column_index]          # get the total occupancies of the columns above our row
                right = memo[i - 1][optimal_column_index + 1]

                if right < up:                                  # if the right column resuls in a lower total occupancy than our current column, go right
                    optimal_column_index += 1

            elif optimal_column_index == m - 1: # far right
                up = memo[i - 1][optimal_column_index]
                left = memo[i - 1][optimal_column_index - 1]

                if left < up:                                   # if the left column resuls in a lower total occupancy than our current column, go left
                    optimal_column_index -= 1

            else:
                up = memo[i - 1][optimal_column_index]
                right = memo[i - 1][optimal_column_index + 1]
                left = memo[i - 1][optimal_column_index - 1]

                if left < up and left < right:
                    optimal_column_index -= 1
                elif right < up and right < left:
                    optimal_column_index += 1

    sections_location.reverse() # O(n) time comp list reversal

    return([minimum_total_occupancy, sections_location])

    # the final time complexity for our function is O(nm) memo creation + O(nm) dp brute force + O(m) min call + O(m) index call + O(n) back tracking + O(n) list reversal
    # resulting in a final time complexity of O(nm)

    # the final auxilliary space complexity of our function is O(nm) memo space + O(n) output list
    # resulting in a final aux space complexity of O(nm)