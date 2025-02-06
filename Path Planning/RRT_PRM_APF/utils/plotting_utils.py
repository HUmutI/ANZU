import matplotlib.pyplot as plt
import numpy as np

def plot_environment(env, ax=None):
    """Plot static and dynamic obstacles on a given axis object (or create a new one)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, env.size)
    ax.set_ylim(0, env.size)
    ax.grid(True)

    # Plot static obstacles
    for obs in env.static_obstacles:
        circle = plt.Circle(obs["center"], obs["radius"], color='r', alpha=0.5)
        ax.add_patch(circle)

    # Plot dynamic obstacles
    for obs in env.dynamic_obstacles:
        circle = plt.Circle(obs["position"], obs["radius"], color='b', alpha=0.5)
        ax.add_patch(circle)

    return ax
