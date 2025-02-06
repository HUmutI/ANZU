import numpy as np

class Environment:
    def __init__(self, size=500, num_dynamic=5, is_dynamic=True):
        """
        Initializes the environment.

        Parameters:
        -----------
        size : int
            The size of the square environment in which to plan.
        num_dynamic : int
            The number of dynamic (moving) obstacles.
        is_dynamic : bool
            Flag to indicate if obstacles should move over time.
        """
        self.size = size
        self.is_dynamic = is_dynamic
        self.static_obstacles = self._create_static_obstacles()
        self.dynamic_obstacles = self._create_dynamic_obstacles(num_dynamic) if is_dynamic else []

    def _create_static_obstacles(self):
        """Creates a list of fixed circular obstacles."""
        return [
            {"center": (100, 200), "radius": 20},
            {"center": (300, 350), "radius": 30},
            {"center": (400, 150), "radius": 25}
        ]
    
    def _create_dynamic_obstacles(self, num):
        """
        Creates moving circular obstacles at random initial locations
        with random velocities.
        """
        obstacles = []
        for _ in range(num):
            position = np.random.rand(2) * self.size
            velocity = np.random.randn(2) * 2  # random velocity
            obstacles.append({"position": position, "velocity": velocity, "radius": 5})
        return obstacles
    
    def update_dynamic_obstacles(self):
        """Moves the dynamic obstacles based on their velocity (if any)."""
        if not self.is_dynamic:
            return
        for obs in self.dynamic_obstacles:
            obs["position"] += obs["velocity"]
            # Keep obstacles inside boundaries
            obs["position"] = np.clip(obs["position"], 0, self.size)
    
    def check_collision(self, x, y):
        """Checks if a point (x, y) collides with any obstacle (static or dynamic)."""
        # Check static obstacles
        for obs in self.static_obstacles:
            center = np.array(obs["center"])
            if np.linalg.norm(center - np.array([x, y])) < obs["radius"]:
                return True
        
        # Check dynamic obstacles
        for obs in self.dynamic_obstacles:
            if np.linalg.norm(obs["position"] - np.array([x, y])) < obs["radius"]:
                return True
        
        return False
