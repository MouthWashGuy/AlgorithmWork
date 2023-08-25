"""FIT2004 Assignment 2 File"""
__author__ = "Brandon Yong Hoong Tak (32025963)"

# ==================== Queue Related Classes ====================

"""
Class description: Class representing a circular queue data structure, used primarily in BFS
Based of the circular queue implementation provided in FIT1008, modified slightly to use a standard python list/array
"""
class CircularQueue:

    MIN_CAPACITY = 1

    def __init__(self, capacity):
        """
        Function description: Constructor for the queue object. Initializes an array used to store queue elements and sets both the rear and front pointers to 0

        Allocating the space for the array is O(n) for both time and space complexity, where n is the capacity of the queue

        Input:
            capacity: An integer value for the maximum capacity of the queue

        Output: None

        Time complexity: O(n) where n is the capacity of the queue
        Aux space complexity: O(n) where n is the capacity of the queue
        """
        self.arr = [None]*max(self.MIN_CAPACITY, capacity)
        self.length = 0
        self.front = 0
        self.rear = 0

    def __len__(self):
        """
        Function description: Returns the number of items in the queue

        Input: None

        Output: The number of elements in the queue

        Time complexity: O(1)
        Aux space complexity: O(1)
        """
        return self.length
    
    def is_full(self):
        """
        Function description: Checks if the queue has reached its maximum capacity

        Input: None

        Output: True if the queue is full, false otherwise

        Time complexity: O(1)
        Aux space complexity: O(1)
        """
        return len(self.arr) == self.length
    
    def is_empty(self):
        """
        Function description: Checks if the queue is empty

        Input: None

        Output: True if the queue is empty, false otherwise

        Time complexity: O(1)
        Aux space complexity: O(1)
        """
        return self.length == 0
    
    def enqueue(self, item):
        """
        Function description: Inserts an item into the back of the queue. Since we are using a circular queue, insertion is O(1) as we are simply manipulating pointer in a static array

        Input: 
            item: The object to be inserted into the queue

        Output: None

        Time complexity: O(1)
        Aux space complexity: O(1)
        """
        if self.is_full():
            raise Exception("Queue is full")
        
        self.arr[self.rear] = item
        self.length += 1
        self.rear = (self.rear + 1) % len(self.arr)

    def dequeue(self):
        """
        Function description: Pops an item from the front of the queue. Since we are using a circular queue, serveing is O(1) as we are simply manipulating pointer in a static array

        Input: 

        Output: The object at the front of the queue

        Time complexity: O(1)
        Aux space complexity: O(1)
        """
        if self.is_empty(): 
            raise Exception("Queue is empty")
        
        self.length -= 1
        item = self.arr[self.front] 
        self.front = (self.front + 1) % len(self.arr)
        return item

# ==================== Graph Related Classes ====================

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

"""
Class description: Class representing a vertex object.
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

        # traversal related
        self.previous = None
        self.visited = False
        self.edge_taken = None
    
"""
Class description: Class representing an edge object with a start, end and weigth value.
Based of the graph implementation shown in the FIT2004 lecture videos provided by Dr Lim Wern Han
"""
class Edge:

    def __init__(self, start, end, capacity):
        """
        Function description: Constructor for the edge object.
        Sets the objects attributes to their appropriate values.

        Input:
            start: The source vertex object (from)
            end: The destination vertex object (to)
            capacity: The capacity of the edge

        Output: None

        Time complexity: O(1)
        Aux space complexity: O(1)
        """
        self.start = start
        self.end = end
        self.capacity = capacity

        # for ford fulkerson, we keep a reference to its parallel edge
        self.associated_edge = None
    
# ==================== Q1 Code ====================

def maxThroughput(connections, maxIn, maxOut, origin, targets):
    """
    1) Fast Backups (10 marks)

    Function description: Given a list of tuples representing the connections between data centres as well the maximum output and input capacities of each data centre, this algorithm computes the maximum flow
    of data that can be achieved from the origin data centre to target data centres by using the Ford Fulkerson method.

    Approach description: To begin, we shall model this problem as a network flow graph where each data centre represents a node and each connection represents an edge with its associated capacity.
    The trick that allows us to account for each data centres maximum incoming and outgoing capacity is to split each data centre into 3 nodes respectively, one incoming node which only has incoming edges,
    an outgoing node which only has outgoing edges, and a middle node which connects the incoming node to the outgoing node using edges with the maxIn and maxOut capacities respectively. This ensures that any flow passing
    through our data centre is constrained by the maximum incoming and outgoing data throughput it can handle. Once we have duplicated our nodes and connected them appropriately we will then connect our origins middle node
    to an extra source node we have made, and each targets middle node to an extra sink node. This ensures that our source has no incoming edges and solves the problem of having multiple targets as well.
    Now that we have finished processing our input into a usable network flow graph, we can simply build a residual network from it and perform the ford fulkerson method in order to compute our maximum flow
    from the source to the sink.

    The overall time complexity of our algorithm can be found by analyzing the time complexity of the 3 major steps of our algorithm. 
    Step 1: Processing input
    Here we create our empty network flow graph O(3D + 2) before populating it by looping over our connections list O(C), our data centres O(D), and our targets O(D) 
    resulting in a final time complexity of O(D + C) which is the complexity required to create an adjacency list based graph
    Step 2: Creating our residual network
    Copying everything over from our network flow graph costs O(D + C), we also end up doubling our edges/connections to accomodate for reverse flow, however this does not effect the overall
    time or space complexity as we are multiplying by a constant factor
    Step 3: Run ford fulkerson on our residual network
    Since we use a BFS in this section, we get an overall time complexity of O(D*C^2)
    This results in an overall time complexity of O(D + C) + O(D + C) + O(D*C^2) which is nothing but O(D*C^2) asymptotically

    aux space wise, we create a network flow graph and a residual network, both of which occupy O(D + C) auxilliary space respectively.
    This results in a final aux space complexity of O(D + C)

    Input:
       connections: A list of tuples representing the connections between each data centre and the associated maximum throughput of the channel as a positive integer
       maxIn: A list of integers representing the maximum incoming data throughput into a node/data centre
       maxOut: A list of integers representing the maximum outgoing data throughput from a node/data centre
       origin: An integer representing the id of the data centre we wish to backup
       target: A list of integers representing the ids of the data centre we wish to send the backups to

    Output: The maximum throughput from the origin to the targets as an integer

    Time complexity: O(D*C^2) where D is the number of data centre/vertexes and C is the number of communication channels/edges
    Aux space complexity: O(D + C) where D is the number of data centre/vertexes and C is the number of communication channels/edges
    """
    
    # To begin lets start processing our input data into something we can use
    # ideally, we want to create a network flow graph from our data which we can easily make a residual network from later
    # since our strategy involves splitting each node into three, we can use an offset to standardize the id of our nodes
    # so, for any given data centre, we split it into an incoming node with id of n, and an outgoing node with id n + 2D while the middle node will have an id of n + D

    offset = len(maxIn) # offset = D

    # Step 1, lets create a network flow graph with an adjacency list of size 3D + 2 where D is the number of data centres
    # the 3D comes from the fact we must split each node/data centre into an incoming node, an outgoing node and the original node
    # the + 2 comes from the need to include a source and a sink as our origin node does not necesarrily have 0 incoming edges and our targets may also have outgoing edges
    # by introducing our own source and sink we can utilize ford fulkersons method effectively
    
    network_flow = Graph(3*offset + 2) # this creates a graph of size 3D + 2, which is O(D) time and space complexity wise

    # Step 2, now lets handle the basic connections first
    for connection in connections: # O(C) time comp and space comp since we make an edge for each connection

        # unpack the values for clarity
        start_vertex = network_flow.vertices[connection[0] + 2*offset] # this is the start, so grab the data centres outgoing node
        end_vertex = network_flow.vertices[connection[1]]              # this is the destination, so grab the data centres incoming node
        capacity = connection[2]

        # now create the edge
        start_vertex.edges.append(Edge(start_vertex, end_vertex, capacity))

    # Step 3, excellent now that all our data centre connections are settled, lets link up each data centres incoming node to its middle node and then its outgoing node respectively
    for i in range(offset): # O(D) time and space comp as we make 2 edges for each vertex

        # unpack the values for clarity
        start_vertex = network_flow.vertices[i] # get the incoming node
        end_vertex = network_flow.vertices[i + offset] # get the associated outgoing node
        capacity = maxIn[i]

        # create the edge
        start_vertex.edges.append(Edge(start_vertex, end_vertex, capacity))

        # unpack the values for clarity
        start_vertex = network_flow.vertices[i + offset] # get the incoming node
        end_vertex = network_flow.vertices[i + 2*offset] # get the associated outgoing node
        capacity = maxOut[i]
        
        # create the edge
        start_vertex.edges.append(Edge(start_vertex, end_vertex, capacity))

    # Step 4, great now all we have to do is link out source to origin and sink to target
    # this bits a bit tricky though, cause we cant just take the minimum of maxIn or maxOut to set the edges capacity
    # since the origin data centre is what generates our flow, we have to ensure our source node is connected to its outgoing node using its maxOut capacity
    # similarly, our target data centres need to link to the sink with their respective maxIn capacities from their incoming nodes

    # for the origin
    start_vertex = network_flow.vertices[3*offset] # heres the source
    end_vertex = network_flow.vertices[origin + offset] # and heres our origins outgoing node
    capacity = maxOut[origin]

    # create the edge
    start_vertex.edges.append(Edge(start_vertex, end_vertex, capacity))

    # for the targets
    for target in targets:  #O(t), where t is the number of targets, t is always < D so this is also O(D) time and space complexity wise

        # unpack for clarity
        start_vertex = network_flow.vertices[target + offset] # heres our targets incoming node
        end_vertex = network_flow.vertices[3*offset + 1] # and heres our sink
        capacity = maxIn[target]

        # create the edge
        start_vertex.edges.append(Edge(start_vertex, end_vertex, capacity))
        
    # Step 5, awesome our network flow graph is done, now we can make a residual network from it
    # this does not delete/replace the previous network flow graph, we could but we wont since this doesnt effect the asymptotic complexity, and is usually kept around in practice
    # now we can run our basic max flow algorithm until we arrive at an augment of 0
    # once that happens, we know we have hit our max flow! \0/
    # rounding up the complexity of all our operations up to this point we have
    # time complexity = O(D) + O(C) + O(D) + O(D) = O(D + C)
    # space complexity = O(D) + O(C) + O(D) + O(D) = O(D + C)

    flow = 0
    residual_network = ResidualNetwork(network_flow) # making this also costs O(D + C) time and space complexity

    # the first do block
    residual_network.BFS(residual_network.vertices[3*offset]) # start from the source and find the shortest path to each node, O(D + C) time complexity and O(D) space
    path = residual_network.getAugmentingPath(residual_network.vertices[3*offset + 1]) # back track from the sink to get the shortest path to the source, O(D) time and space complexity
    augment = residual_network.getAugmentAmount(path) # iterate over the path to find its bottle neck amount, O(D) time complexity
    residual_network.augmentNetwork(path, augment) # update the edges along the path in the residual network, O(D) time complexity
    flow += augment # add the augment to the flow

    # now repeat the same process until we receive an augment of 0, indicating we have achieved the maximum flow
    # since this loop can run up to O(F) times, where F is the maximum flow of the network, and we search for a path from source to sink each time, the complexity of this operation is O(C*F)
    # however since we use a BFS to search, which always returns the shortest path from source to sink, thus reducing the probability of having a small bottleneck,
    # we end up with a final time complexity of O(4D*C^2) = O(D*C^2)
    while augment > 0:
        residual_network.BFS(residual_network.vertices[3*offset])
        path = residual_network.getAugmentingPath(residual_network.vertices[3*offset + 1])
        augment = residual_network.getAugmentAmount(path)
        residual_network.augmentNetwork(path, augment)
        flow += augment

    return flow # return the max flow

    # tallying our final complexity we have
    # time complexity = O(D + C) + O(D + C) + O(D*C^2) = O(D*C^2)
    # space complexity = O(D + C) + O(D + C) + O(D) = O(D + C)

"""
Class description: Class representing a residual network used in the Ford Fulkerson method
Loosely inspired by the ford fulkerson implementation shown in the FIT2004 lecture videos provided by Dr Lim Wern Han
"""
class ResidualNetwork(Graph) :

    def __init__(self, network_flow):
        """
        Function description: Constructor for an adjacency list based residual network object. Inherits from the graph class and makes use of its super constructor to 
        initialize an array of equal size when fed a network flow graph. All vertices are then copied over from the network flow graph and additional edges are added

        Allocating the space for the array is O(V) where V is the number of vertices in the network flow graph used to construct the residual network
        copying the entire adjacency list over costs O(V + E) as we have to iterate over each vertex and its edge list

        Input:
            network_flow: A network flow graph

        Output: None

        Time complexity: O(V + E) where V and E are the total vertices and edges in the residual network
        Aux space complexity: O(V + E) where V and E are the total vertices and edges in the residual network
        """
        # lets make that adjacency list again
        super().__init__(len(network_flow.vertices)) # O(V)

        # copy everything over from the network flow to the residual, this is O(V + E)
        for vertex in network_flow.vertices:
            for edge in vertex.edges:

                # unpack for clarity
                start_vertex = self.vertices[edge.start.id]
                end_vertex = self.vertices[edge.end.id]
                capacity = edge.capacity

                # heres the magic, we create a backwards and forward edge for each and provide them a reference to each other
                forward_edge = Edge(start_vertex, end_vertex, capacity)
                backward_edge = Edge(end_vertex, start_vertex, 0)
                forward_edge.associated_edge = backward_edge
                backward_edge.associated_edge = forward_edge
            
                # now append them to the correct vertex
                # technically this makes O(V + 2E) but that is the same as O(V + E) asymptotically
                start_vertex.edges.append(forward_edge)
                end_vertex.edges.append(backward_edge)

    def BFS(self, start):
        """
        Function description: A generic BFS used in the ford fulkerson algorithm. Based of the bfs pseudocode shown in the FIT2004 course notes provided by Daniel Anderson.
        Slight modifications have been made to store important data within vertexes for backtracking purposes

        Input:
            start: The node to start searching from

        Output: None

        Time complexity: O(V + E) where V and E are the total vertices and edges in the residual network
        Aux space complexity: O(V) where V is the total number of vertices in the residual network
        """
        for vertex in self.vertices: # reset all node attibutes O(V)
            vertex.visited = False
            vertex.previous = None
            vertex.edge_taken = None

        start.visited = True
        queue = CircularQueue(len(self.vertices)) # create a queue and put the starting node into it, this can grow to O(V) space complexity
        queue.enqueue(start)

        while len(queue) > 0: # O(V + E) time complexity
            curr_node = queue.dequeue() # serve the node at the front of the queue

            for edge in curr_node.edges:   # for each of its edges
                if edge.capacity > 0:      # if there is still capacity left to flow
                    neighbour = edge.end   # take the edge to its neighbour 
                    if neighbour.visited == False: # update all its attributes accordingly
                        neighbour.visited = True
                        neighbour.previous = curr_node
                        neighbour.edge_taken = edge
                        queue.enqueue(neighbour) # add the neighbour to the queue

    def getAugmentingPath(self, start):
        """
        Function description: A simple backtracking algorithm that backtracks from a given vertex to the original BFS source.
        Appends the edges taken to a list and returns it

        Input:
            start: The node to start backtracking from

        Output: A list of edges taken from the bfs source to our starting node in reverse order, ordering in this case is not important

        Time complexity: O(V) where V is the total number of vertices in the residual network
        Aux space complexity: O(V) where V is the total number of vertices in the residual network
        """
        path = [] # O(V) space comp
        curr_node = start

        while curr_node.previous != None: # O(V) time comp
            path.append(curr_node.edge_taken)
            curr_node = curr_node.previous

        return path
    
    def getAugmentAmount(self, path):
        """
        Function description: A simple function used to find the bottleneck/smallest capacity found along a given path of edges
        even though we provide a path of edges, it is still constrained by O(V) as we have exactly one edges between each vertex along the path so technically O(V - 1)

        Input:
            path: A list of edges

        Output: The smallest capacity among the provided edges

        Time complexity: O(V) where V is the total number of vertices in the residual network
        Aux space complexity: O(1)
        """
        if len(path) == 0:
            return 0
        else:
            minimum_edge = min(path, key=lambda edge:edge.capacity) # fancy lambda function that gets the edges capacity instead of the edge itself, min call is O(V)
            return minimum_edge.capacity
    
    def augmentNetwork(self, path, augment):
        """
        Function description: A function that updates the capacities of the edges along our path as well as their associated edges

        Input:
            path: A list of edges
            augment: The amount to update each edges capacity by

        Output: None

        Time complexity: O(V) where V is the total number of vertices in the residual network
        Aux space complexity: O(1)
        """
        for edge in path: # O(V)
            edge.capacity -= augment
            edge.associated_edge.capacity += augment 

# ==================== Q2 Code ====================

"""
Class description: Class representing a trie node used in the trie data structure
Based of the node implementation shown in the FIT2004 lecture videos provided by Dr Lim Wern Han
"""
class TrieNode:

    def __init__(self):
        """
        Function description: Constructor for the Trie Node object. 
        Initializes an array used to store links to other Trie node objects. Since the english language consists of 26 characters and we must include our terminal character $,
        we end up with an array of size 27

        Input:

        Output: None

        Time complexity: O(1)
        Aux space complexity: O(1)
        """
        self.links = [None] * 27

        # auto complete related
        self.frequency = 0
        self.implicit_link = float('inf') 

"""
Class description: Class representing a trie data structure
Based of the trie implementation shown in the FIT2004 lecture videos provided by Dr Lim Wern Han
"""
class CatsTrie:
    """
    2) catGPT (10 marks)

    Approach description: The trick to ensuring our autocomplete function runs within complexity bounds is to store some important data at each Trie node within our Trie. Namely, we want to be able to know
    at any given node, what is the index of the next node that leads us to the terminal node containing the highest frequency. We can do this during our insertion process which we will implement recursively.
    The process goes as follows, as we recurse downwards, we create or traverse to the nodes associated with the characters in the string we are inserting, when we finally reach our base case (terminal node), we get the value
    of its frequency attribute and then begin moving upwards as we exit the recursion. At each level in the call stack as we go upwards, we will perform a check on that levels curr node:

    is curr node frequency < previous node frequency?

    if it is then we know we are returning from inserting a word with a higher frequency, so we must update that nodes frequency to match and then update its implicit link attribute with the previous nodes index.
    In doing so, the next time we walk down the trie, no matter where we stop, we always know the index of the next node that leads us to the terminal node containing the highest frequency

    In the case of frequency ties however, we can simply compare the current nodes implicit link attribute to the previous node, if we find that the current nodes is bigger, then we know that we are
    returning from inserting a word that is lexicographically smaller than the previous word we inserted, hence update the nodes attributes.

    The autocomplete function itself becomes trivial after this, as most of the work has been front loaded into our insert method.
    To autocomplete any given sentence, simply walk down the trie based on the given prompt, and wherever you stop, use the implicit pointer of that node
    to find the next node that brings you towards the terminal node containing the highest frequency word and so on. 
    """
    def __init__(self, sentences):
        """
        Function description: Constructor for the Trie object. Creates the root node before inserting each string from sentences into the trie.
        also updates the root nodes frequency and implicit link to point to the string path that has the highest frequecy in lexicographical order.

        Since we insert N strings and each inserts costs M, where M is the length of the string. 
        We end up with a time complexity of O(NM), where M is the length of the longest string.

        In the worst case where each string does not share any common letters, the trie itself can contain NM nodes resulting in an auxilliary space complexity of O(NM)

        Input:
            sentences: A list of strings to insert into the trie

        Output: None

        Time complexity: O(NM), where N is the number of strings in sentences and M is the length of the longest string
        Aux space complexity: O(NM), where N is the number of strings in sentences and M is the length of the longest string
        """
        self.root = TrieNode() # create the root node

        curr_node = self.root
        
        # insert each sentence recursively
        for sentence in sentences: # O(N)
            frequency, implicit_link = self.insert(sentence, curr_node, 0) # O(M)

            # if the word we just inserted has a higher frequency or (same frequency and smaller lexicographical order)
            # update the root nodes frequency and implicit link to accomodate
            if frequency > curr_node.frequency or (frequency == curr_node.frequency and implicit_link < curr_node.implicit_link):
                curr_node.frequency = frequency
                curr_node.implicit_link = implicit_link

    def insert(self, string, curr_node, char_pointer):
        """
        Function description: Recursive algorithm that inserts a sentence into the trie character by character. 
        The key to allowing us to perform our autocomplete efficiently is to update the attributes of our Trie Nodes on the way up from recursion.

        We at most recurse M times before hitting our base case, where M is the number of characters in the string. Hence we have a time and aux space complexity of O(M)

        Input:
            string: The string to insert
            curr_node: The current node we are at 
            char_pointer: A pointer indicating which character in the string we are inserting

        Output: 
            frequency:     The number of times this particular word has been inserted, while returning upwards from recursion the result of this will propagate upwards
                           along each Trie Node in this strings path.
            implicit_link: A pointer pointing to which Trie Node to traverse to next when asked to autocomplete, this is kept up to date during insertion operations and prioritizes
                           string paths with higher frequency or those with the same frequency but a lower lexicographical value
             

        Time complexity: O(M), where M is the length of the string
        Aux space complexity: O(M), where M is the length of the string
        """
        # base case
        # when our character pointer is equal to the length of our string, we know that the word is complete
        if char_pointer == len(string):

            index = 0 # we know that this is the terminal character

            if curr_node.links[index] == None:
                curr_node.links[index] = TrieNode() # the first time we insert a word

            curr_node = curr_node.links[index]

            curr_node.frequency += 1 # add one to its frequency

            # now return the frequency and index respectively
            return (curr_node.frequency, index)
        
        # recursive case
        # if we havent reached the bottom yet, keep going
        else:

            # on the way down logic
            index = ord(string[char_pointer]) - 96
            if curr_node.links[index] == None:
                curr_node.links[index] = TrieNode() # the first time we insert a letter at this node
            curr_node = curr_node.links[index]

            # recurse
            frequency, implicit_link = self.insert(string, curr_node, char_pointer + 1)

            # on the way up logic
            # our recursion should always provide us with the fequency and implicit link of the node below us we just inserted
            # now we can perform a check to see if we should override this current nodes frequency and link based on the result of the lower nodes
            # if we find that the word we just inserted along this path has a higher frequency, or the same but with a lower lexicographical value, override this nodes previous attributes
            if frequency > curr_node.frequency or (frequency == curr_node.frequency and implicit_link < curr_node.implicit_link):
                curr_node.frequency = frequency
                curr_node.implicit_link = implicit_link

            return (curr_node.frequency, index)
    
    def autoComplete(self, prompt):
        """
        Function description: Recursive algorithm that inserts a sentence into the trie character by character. 
        The key to allowing us to perform our autocomplete efficiently is to update the attributes of our Trie Nodes on the way up from recursion.

        We at most recurse M times before hitting our base case, where M is the number of characters in the string. Hence we have a time and aux space complexity of O(M)

        Input:
            prompt: The string to autcomplete

        Output: The fully auto-completed string

        Time complexity: O(X + Y), where X is the length of the prompt and Y is the length of the most frequent sentence in sentences that begins with the prompt
        Aux space complexity: O(X + Y), where X is the length of the prompt and Y is the length of the most frequent sentence in sentences that begins with the prompt
        """
        output = [] # can grow up to X or up to Y if it exists, hence O(X + Y) aux space

        # start at the root
        curr_node = self.root

        for char in prompt: # O(X) time complexity
            
            output.append(char)
            index = ord(char) - 96

            # walk down the trie until we reach the end of the prompt, can exit here if the sentence doesnt exist resulting in O(X)
            if curr_node.links[index] == None:
                return None
            else:
                curr_node = curr_node.links[index]

        # if we have not exited the function yet, then we know there is a valid sentence that can be completed
        # now that we are at the end of the prompt, we just need to keep going to reach the correct terminal
        # we know exactly where to go since we saved an implicit link at every Trie Node. Hence, no matter where we stop, the node always knows the way
        # to the next node on the path to the sentence with the highest frequency
        while curr_node.implicit_link > 0: # Worst case is starting from root, this can loop O(Y) times to reach the autocompleted sentence
            output.append(chr(curr_node.implicit_link + 96))
            curr_node = curr_node.links[curr_node.implicit_link]

        return "".join(output) # joining here is also O(Y)
    
    # tallying our final complexity we have
    # time complexity = O(X + 2Y) = O (X + Y)
    # space complexity = O(X + Y)

# ==================== Main Functions ====================

if __name__ == "__main__":

    sentences = ["abc", "a", "abc", "aaa", "aaa"]
    mycattrie = CatsTrie(sentences)
    mycattrie.autoComplete("ab")

