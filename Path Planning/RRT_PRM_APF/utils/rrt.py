import random
import numpy as np

class RRT:
    def __init__(self, env, start, goal, step_size=10, max_iter=500):
        """
        RRT motion planning class.

        Parameters:
        -----------
        env : Environment
            An instance of the Environment class.
        start : tuple (float, float)
            (x, y) for the starting position.
        goal : tuple (float, float)
            (x, y) for the goal position.
        step_size : float
            The incremental distance in each step of expansion.
        max_iter : int
            The maximum number of iterations for the RRT.
        """
        self.env = env
        self.start = start
        self.goal = goal
        self.step_size = step_size
        self.max_iter = max_iter

        # Data structures for the RRT
        self.tree = [start]            # List of nodes
        self.parent = {start: None}    # Parent dictionary for path extraction

    def _get_random_point(self):
        """Randomly sample a point in the environment."""
        return (random.uniform(0, self.env.size), random.uniform(0, self.env.size))

    def _nearest_node(self, point):
        """Find the nearest node in the tree to the given point."""
        return min(self.tree, key=lambda n: np.linalg.norm(np.array(n) - np.array(point)))

    def _steer(self, from_node, to_point):
        """Steers from one node towards another point, returning the new node."""
        vector = np.array(to_point) - np.array(from_node)
        dist = np.linalg.norm(vector)
        if dist < self.step_size:
            return to_point
        new_point = np.array(from_node) + (vector / dist) * self.step_size
        return tuple(new_point)

    def plan(self):
        """
        Plans a path using the RRT algorithm.

        Returns:
        --------
        path : list of tuples
            The found path from start to goal, or None if no path is found.
        """
        for _ in range(self.max_iter):
            rand_point = self._get_random_point()
            nearest = self._nearest_node(rand_point)
            new_node = self._steer(nearest, rand_point)

            # Check for collision
            if not self.env.check_collision(new_node[0], new_node[1]):
                self.tree.append(new_node)
                self.parent[new_node] = nearest

                # Check if we've reached/are close to the goal
                if np.linalg.norm(np.array(new_node) - np.array(self.goal)) < self.step_size:
                    return self._extract_path(new_node)

        return None  # No path found

    def _extract_path(self, end_node):
        """Backtracks from the goal to the start to extract the path."""
        path = [end_node]
        while path[-1] in self.parent:
            p = self.parent[path[-1]]
            if p is None:
                break
            path.append(p)
        return path[::-1]

    def get_rrt_tree(self):
        """Optional: returns the tree nodes and parents if you want to plot them externally."""
        return self.tree, self.parent
