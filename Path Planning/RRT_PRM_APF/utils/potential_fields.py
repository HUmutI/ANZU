import numpy as np

class PotentialFields:
    def __init__(self, env, start, goal, step_size=0.05, max_steps=1000):
        self.env = env
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.step_size = step_size
        self.max_steps = max_steps

        # Tuning parameters for potential fields (example values)
        self.k_att = 1.0    # Attractive constant
        self.k_rep = 10000000.0    # Repulsive constant
        self.influence_radius = 200.0  # Range of obstacle repulsion

    def plan(self):
        """Perform a gradient-descent-like approach based on artificial potentials."""
        path = [self.start.copy()]
        current = self.start.copy()

        for _ in range(self.max_steps):
            grad = self._compute_gradient(current)
            # move in the direction opposite the gradient (steepest descent)
            new_pos = current - self.step_size * grad
            # Check for collisions
            if not self.env.check_collision(new_pos[0], new_pos[1]):
                current = new_pos
                path.append(current.copy())
            else:
                # If a direct step hits collision, you might try smaller steps or alternative strategies
                return None

            # Check if close enough to goal
            if np.linalg.norm(current - self.goal) < 10.0:
                path.append(self.goal.copy())
                return [tuple(p) for p in path]
        
        return None  # Gave up

    def _compute_gradient(self, position):
        """Compute the total potential gradient at 'position' = grad(U_att + U_rep)."""
        # Attractive gradient
        grad_att = self.k_att * (position - self.goal)

        # Repulsive gradient
        grad_rep = np.zeros_like(position)
        for obs in self.env.static_obstacles:
            obs_center = np.array(obs["center"])
            dist = np.linalg.norm(position - obs_center)
            if dist < self.influence_radius:
                # partial derivative of the repulsive potential
                rep_dir = (position - obs_center) / dist  # direction from obstacle
                grad_rep += self.k_rep * (1.0 / dist - 1.0 / self.influence_radius) * (1.0 / dist**2) * rep_dir
        for obs in self.env.dynamic_obstacles:
            obs_center = obs["position"]
            dist = np.linalg.norm(position - obs_center)
            if dist < self.influence_radius:
                rep_dir = (position - obs_center) / dist
                grad_rep += self.k_rep * (1.0 / dist - 1.0 / self.influence_radius) * (1.0 / dist**2) * rep_dir
        
        total = (grad_att - grad_rep) / np.linalg.norm(grad_att - grad_rep)
        return total * 100
