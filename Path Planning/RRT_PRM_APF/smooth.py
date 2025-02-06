# Smooth the path using cubic spline interpolation
# Check main.py

import time
import matplotlib.pyplot as plt

from utils.environment import Environment
from utils.rrt import RRT
from utils.prm import PRM
from utils.potential_fields import PotentialFields
from utils.plotting_utils import plot_environment
from utils.smooth_path import smooth_path


# 1) Create the dynamic environment
env = Environment(size=500, num_dynamic=3, is_dynamic=True)

# 2) Start and goal
start = (50, 50)
goal = (450, 450)

# 3) List of planners you want to test
planners = {
    "RRT": RRT(env, start, goal, step_size=15, max_iter=1000),
    "PRM": PRM(env, start, goal, num_samples=200, connection_radius=70),
    "PotentialFields": PotentialFields(env, start, goal, step_size=5, max_steps=500)
}

for name, planner in planners.items():
    print(f"--- Running {name} ---")
    start_time = time.time()
    path = planner.plan()
    end_time = time.time()
    duration = end_time - start_time

    if path is not None:
        print(f"{name} found a path in {duration:.4f} seconds!")
    else:
        print(f"{name} failed to find a path in {duration:.4f} seconds!")

    # Smooth the path
    if path is not None:
        smoothed_path = smooth_path(path)

    # Plot the environment and result
    fig, ax = plt.subplots(figsize=(6,6))
    ax = plot_environment(env, ax=ax)
    # Plot start/goal
    ax.scatter(start[0], start[1], c='g', s=50, label='Start')
    ax.scatter(goal[0], goal[1], c='b', s=50, label='Goal')

    if path is not None:
        # path is a list of (x, y)
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, 'r--', linewidth=2, label=f'{name} Path (Original)')
    
        smoothed_x = [p[0] for p in smoothed_path]
        smoothed_y = [p[1] for p in smoothed_path]
        ax.plot(smoothed_x, smoothed_y, 'g-', linewidth=2, label=f'{name} Path (Smoothed)')

    ax.set_title(f"{name} Result")
    ax.legend()
    plt.show()
    
"""
# 4) If environment is dynamic, we can demonstrate updates and re-running planners
if env.is_dynamic:
    for i in range(3):  # just do 3 dynamic updates for demo
        print(f"\n--- Dynamic update step {i+1} ---")
        env.update_dynamic_obstacles()

        for name, planner_class in [("RRT", RRT), ("PRM", PRM), ("PotentialFields", PotentialFields)]:
            print(f"\nRe-planning with {name} after dynamic update:")
            # Create a fresh planner with the updated environment
            planner = planner_class(env, start, goal)
            path = planner.plan()
            if path is not None:
                print(f"{name} found a path!")
            else:
                print(f"{name} did NOT find a path!")

        # Plot environment after each update if desired
        fig, ax = plt.subplots(figsize=(6,6))
        ax = plot_environment(env, ax=ax)
        ax.scatter(start[0], start[1], c='g', s=50, label='Start')
        ax.scatter(goal[0], goal[1], c='b', s=50, label='Goal')
        ax.set_title(f"Dynamic Environment After Update {i+1}")
        ax.legend()
        plt.show()
"""