import numpy as np
import random

class PRM:
    def __init__(self, env, start, goal, num_samples=200, connection_radius=50):
        """
        PRM motion planning class (very simplified).

        Parameters:
        -----------
        env : Environment
            An instance of the Environment class.
        start : tuple
            (x, y) for the start.
        goal : tuple
            (x, y) for the goal.
        num_samples : int
            How many random free-space samples to pick.
        connection_radius : float
            A distance threshold for connecting two samples.
        """
        self.env = env
        self.start = start
        self.goal = goal
        self.num_samples = num_samples
        self.connection_radius = connection_radius

        self.nodes = []
        self.edges = {}  # adjacency list: node -> list of connected nodes

    def plan(self):
        """
        Builds a PRM roadmap and tries to find a path from start to goal.

        Returns: 
        --------
        path : list of tuples or None
            The path from start to goal, if found.
        """
        # 1. Sample free points
        self._sample_points()

        # 2. Connect neighbors
        self._connect_neighbors()

        # 3. Insert start and goal explicitly
        self._add_start_and_goal()

        # 4. Try to find path in the roadmap
        path = self._search_path()
        return path

    def _sample_points(self):
        """Samples random points in free space and stores them in self.nodes."""
        for _ in range(self.num_samples):
            x = random.uniform(0, self.env.size)
            y = random.uniform(0, self.env.size)
            # check if in free space
            if not self.env.check_collision(x, y):
                self.nodes.append((x, y))

    def _connect_neighbors(self):
        """Tries to connect each node with others within connection_radius if no collision."""
        for i, nodeA in enumerate(self.nodes):
            for j, nodeB in enumerate(self.nodes[i+1:], start=i+1):
                if np.linalg.norm(np.array(nodeA) - np.array(nodeB)) < self.connection_radius:
                    # We do not check the line segment for collisions in this skeleton
                    # but a real PRM would do a segment-check for collisions
                    if self._safe_to_connect(nodeA, nodeB):
                        self._add_edge(nodeA, nodeB)
                        self._add_edge(nodeB, nodeA)

    def _safe_to_connect(self, A, B):
        """Checks if the straight line from A to B collides with obstacles (simplified)."""
        # This is where you'd implement a function to check every few steps along the segment
        # For brevity, we'll skip that here or do a minimal check.
        return True

    def _add_edge(self, nodeA, nodeB):
        if nodeA not in self.edges:
            self.edges[nodeA] = []
        self.edges[nodeA].append(nodeB)

    def _add_start_and_goal(self):
        """Add start & goal to the node list and connect them to the roadmap if possible."""
        self.nodes.append(self.start)
        self.nodes.append(self.goal)

        # Simple connection if within radius
        for node in self.nodes:
            # connect to start
            if np.linalg.norm(np.array(node) - np.array(self.start)) < self.connection_radius:
                if not self.env.check_collision(*node):
                    self._add_edge(self.start, node)
                    self._add_edge(node, self.start)
            # connect to goal
            if np.linalg.norm(np.array(node) - np.array(self.goal)) < self.connection_radius:
                if not self.env.check_collision(*node):
                    self._add_edge(self.goal, node)
                    self._add_edge(node, self.goal)

    def _search_path(self):
        """Use a graph search (e.g., BFS/DFS) to find path from start to goal."""
        from collections import deque

        visited = set()
        queue = deque([(self.start, [self.start])])  # store (node, path_so_far)

        while queue:
            current, path = queue.popleft()
            if current == self.goal:
                return path
            visited.add(current)
            for neighbor in self.edges.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

        return None  # no path found

    def get_roadmap(self):
        """Optional: returns nodes and edges for external plotting."""
        return self.nodes, self.edges
