# 2D Dynamic Potential Field with a moving goal
# Static obstacles
# Minimum turning radius and initial direction is not considered

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

k_att = 2.0
k_rep = 200.0
d0 = 20.0
dt = 0.01
max_rep_force = 300.0

#goal = np.array([10, 10])
#obstacles = np.array([[3, 3.5], [6, 7], [9, 9], [8, 4], [5, 5]])

import numpy as np

# Grid borders 
grid_min, grid_max = -5, 15

# Random(not too much) goal location
goal = np.random.uniform(10, grid_max, size=2)

# Random 3 obstacles
num_obstacles = 3
obstacles = np.random.uniform(0, 10, size=(num_obstacles, 2))

print("Goal:", goal)
print("Obstacles:\n", obstacles)


def attractive_force(position, goal):
    """
    Calculate the attractive force towards the goal
    """
    return -k_att * (position - goal)

def repulsive_force(position, obstacles, d0, k_rep):
    """
    Calculate the repulsive force from obstacles
    """
    force = np.zeros(2)
    for obs in obstacles:
        diff = position - obs
        dist = np.linalg.norm(diff)
        if dist < d0:
            repulsive_force = k_rep * ((1/dist) - (1/d0)) * (1/(dist**2)) * (diff/dist)
            if np.linalg.norm(repulsive_force) > max_rep_force:
                repulsive_force = repulsive_force / np.linalg.norm(repulsive_force) * max_rep_force
            force += repulsive_force
    return force

def total_force(position, goal, obstacles, d0, k_att, k_rep):
    """
    Calculate the total force on the robot
    """
    force_att = attractive_force(position, goal)
    force_rep = repulsive_force(position, obstacles, d0, k_rep)
    return force_att + force_rep

robot_position = np.array([-5, -5])
path_data = [robot_position.copy()]

x_range = np.linspace(-5, 15, 50)
y_range = np.linspace(-5, 15, 50)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)
U = np.zeros_like(X)
V = np.zeros_like(Y)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pos = np.array([X[i, j], Y[i, j]])
        att_potential = 0.5 * k_att * np.linalg.norm(pos - goal)**2
        rep_potential = 0
        for obs in obstacles:
            dist = np.linalg.norm(pos - obs)
            if dist < d0:
                rep_potential += 0.5 * k_rep * ((1/dist) - (1/d0))**2
        Z[i, j] = att_potential + rep_potential
        
        force = total_force(pos, goal, obstacles, d0, k_att, k_rep)
        U[i, j] = force[0]
        V[i, j] = force[1]

fig, ax = plt.subplots()
ax.set_xlim(-5, 15)
ax.set_ylim(-5, 15)

#obstacles_plot, = ax.plot(obstacles[:, 0], obstacles[:, 1], 'ro', label='Obstacles')
import matplotlib.patches as patches

obstacle_radius = 1.0  # Adjust this value to change obstacle size

# Remove old obstacle plotting
obstacles_plot = []  

# Draw obstacles as circles
for obs in obstacles:
    circle = patches.Circle(obs, obstacle_radius, color='red', edgecolor='black', linewidth=2, alpha=1, zorder=10)
    ax.add_patch(circle)
    obstacles_plot.append(circle)

goal_plot, = ax.plot([], [], 'bo', label='Goal')
path_plot, = ax.plot([], [], 'b-', label='Robot Path')
robot_plot, = ax.plot([], [], 'bo')
potential_contour = ax.contourf(X, Y, Z, levels=100, cmap='viridis')
quiver_all = ax.quiver(X, Y, U, V, color='white', alpha=0.5)
quiver_robot = ax.quiver([], [], [], [], color='red')

def init():
    path_plot.set_data([], [])
    robot_plot.set_data([], [])
    goal_plot.set_data([],[])
    quiver_robot.set_UVC([], [])
    return path_plot, robot_plot, goal_plot, quiver_robot

def update(frame):
    global robot_position
    global goal
    force = total_force(robot_position, goal, obstacles, d0, k_att, k_rep)
    #HEDEF NOKTAYI DİNAMİK YAPMAK 
    goal[0] = goal[0] - 0.02
    robot_position = robot_position + force * dt
    path_data.append(robot_position.copy())

    path = np.array(path_data)
    path_plot.set_data(path[:, 0], path[:, 1])
    robot_plot.set_data(robot_position[0], robot_position[1])
    goal_plot.set_data(goal[0],goal[1])

    quiver_robot.set_offsets(robot_position)
    quiver_robot.set_UVC(force[0], force[1])
    return path_plot, robot_plot, quiver_robot, goal_plot

ani = animation.FuncAnimation(fig, update, frames=2000, init_func=init, blit=True, interval=10, repeat=False)

ax.legend(loc='upper left')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Dynamic Potential Field')
ax.grid()

plt.show()
