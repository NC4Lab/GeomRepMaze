"""Useful function for computations on graphs"""

from collections import defaultdict
import math
import sys

def build_graph(edges):
    """builds a graph from a list of edges"""
    graph = defaultdict(list)

    for edge in edges:
        a, b = edge[0], edge[1]

        # Creating the graph as adjacency list
        graph[str(a)].append(b)
        graph[str(b)].append(a)
    return graph




def BFS_SP(graph, start, goal):
    """Breadth first search to find shorted path in a graph"""

    explored = []

    # Queue for traversing the graph in the BFS
    queue = [[start]]

    # If the desired node is reached
    if start == goal:
        #print("Same Node")
        return None

    # Loop to traverse the graphwith the help of the queue
    while queue:
        path = queue.pop(0)
        node = path[-1]

        # Condition to check if thecurrent node is not visited
        if node not in explored:
            neighbours = graph[str(node)]

            # Loop to iterate over the neighbours of the node
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)

                # Condition to check if the neighbour node is the goal
                if neighbour == goal:
                    return new_path
            explored.append(node)

    # Condition when the nodes are not connected
    print("So sorry, but a connecting path doesn't exist :(")
    return None


def compute_path_distance(path):
    """length of path. Path is a list of coordinate (nodes coordinates in euclidean space)"""

    if path is None:
        return 0

    d = 0
    for i in range(len(path)-1):
        d = d + math.dist(path[i], path[i+1]) #TODO add order as arg

    return d


def dijkstra_algorithm(graph, start_node, maze, maze_config):
    unvisited_nodes = list(graph.get_nodes())

    # We'll use this dict to save the cost of visiting each node and update it as we move along the graph
    shortest_path = {}

    # We'll use this dict to save the shortest known path to a node found so far
    previous_nodes = {}

    # We'll use max_value to initialize the "infinity" value of the unvisited nodes
    max_value = sys.maxsize
    for node in unvisited_nodes:
        shortest_path[node] = max_value
    # However, we initialize the starting node's value with 0
    shortest_path[start_node] = 0

    # The algorithm executes until we visit all nodes
    while unvisited_nodes:
        # The code block below finds the node with the lowest score
        current_min_node = None
        for node in unvisited_nodes:  # Iterate over the nodes
            if current_min_node == None:
                current_min_node = node
            elif shortest_path[node] < shortest_path[current_min_node]:
                current_min_node = node

        # The code block below retrieves the current node's neighbors and updates their distances
        neighbors = graph.get_outgoing_edges(current_min_node)
        for neighbor in neighbors:
            tentative_value = shortest_path[current_min_node] + graph.value(current_min_node, neighbor)
            if tentative_value < shortest_path[neighbor]:
                shortest_path[neighbor] = tentative_value
                # We also update the best path to the current node
                previous_nodes[neighbor] = current_min_node

        # After visiting its neighbors, we mark the node as "visited"
        unvisited_nodes.remove(current_min_node)

    return previous_nodes, shortest_path
