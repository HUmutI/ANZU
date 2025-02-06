from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np

def smooth_path(path):
    """
    Apply cubic spline interpolation to smooth the path.
    
    Parameters:
    -----------
    path : list of tuples
        The list of waypoints representing the path.
    
    Returns:
    --------
    smoothed_path : list of tuples
        The smoothed path as a list of waypoints.
    """
    # Extract x and y coordinates from the path
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]

    # Create a cubic spline interpolator for both x and y
    cs_x = CubicSpline(range(len(path_x)), path_x)
    cs_y = CubicSpline(range(len(path_y)), path_y)

    # Define a finer resolution for the smooth path
    finer_resolution = np.linspace(0, len(path_x) - 1, num=500)

    # Generate the smoothed path
    smoothed_x = cs_x(finer_resolution)
    smoothed_y = cs_y(finer_resolution)

    # Combine the smoothed x and y coordinates into a list of points
    smoothed_path = list(zip(smoothed_x, smoothed_y))

    return smoothed_path

def plot_smoothed_path(path, smoothed_path):
    """
    Plots the original path and the smoothed path.
    """
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]

    smoothed_x = [p[0] for p in smoothed_path]
    smoothed_y = [p[1] for p in smoothed_path]

    plt.figure(figsize=(8, 8))
    plt.plot(path_x, path_y, 'r--', label="Original Path")
    plt.plot(smoothed_x, smoothed_y, 'g-', label="Smoothed Path")
    plt.title("Path Smoothing Example")
    plt.legend()
    plt.grid(True)
    plt.show()
