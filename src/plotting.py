# Plotting funcs for the project

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from datetime import datetime

def save_plot_with_timestamp(fig, base_filename, folder="plots"):
    # Create folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    # Get current timestamp in format YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Construct full filename with timestamp
    filename = f"{base_filename}_{timestamp}.png"
    filepath = os.path.join(folder, filename)
    
    # Save figure
    fig.savefig(filepath, dpi=150)
    print(f"Saved plot to {filepath}")

def plot_rewards(reward_list):
    fig = plt.figure(figsize=(12, 6))
    reward_list = np.array(reward_list)
    new_reward_list = reward_list[reward_list > -1600]
    plt.plot(new_reward_list, linestyle="dashed", marker="o", markersize=3,label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    save_plot_with_timestamp(fig, "rewards")
    plt.close(fig)
    #plt.show()

# Grid resolution and range
grid_size = 100  # more = smoother, but slower
x = np.linspace(-1, 1, grid_size)
y = np.linspace(-1, 1, grid_size)
X, Y = np.meshgrid(x, y)  # shape: (grid_size, grid_size)

def potential_fn(x, y):
    rho = np.sqrt(x**2 + y**2)

    # Vectorized mask
    U_hat = np.zeros_like(rho)
    mask = rho <= 0.5
    U_hat[mask] = 16 * 0.25 * (rho[mask]**2 - 0.25)**2

    U_vortex = 0.1 * np.arctan2(-x, y)

    return U_hat + U_vortex


def plot_all_trajectories(trajectories, potential_fn=None):
    fig, ax = plt.subplots(figsize=(8, 8))

    if potential_fn:
        xx, yy = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        zz = potential_fn(xx, yy)
        cf = ax.contourf(xx, yy, zz, levels=50, cmap="viridis", alpha=0.7)
        cbar_potential = fig.colorbar(cf, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
        cbar_potential.set_label('Potential')

    cmap = plt.get_cmap("plasma")
    n = len(trajectories)

    for i in range(0, n, 10):
        traj = np.array(trajectories[i])
        x, y = traj[:, 0], traj[:, 1]
        color = cmap(i / (n - 1))
        ax.plot(x, y, color=color, linewidth=1.5, alpha=0.9)

    # Add start and goal points
    start = np.array([-0.5, 0.0])
    goal = np.array([0.5, 0.0])
    ax.plot(*start, 'ro', markersize=10, label='Start')
    ax.plot(*goal, 'ro', markersize=10, label='Goal')
    ax.text(*start + np.array([0.05, -0.1]), 'Start', color='black', fontsize=12)
    ax.text(*goal + np.array([-0.15, -0.1]), 'Goal', color='black', fontsize=12)

    ax.set_title("Agent Trajectories Over Episodes")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal', 'box')

    norm = mcolors.Normalize(vmin=0, vmax=n-1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar_trajectory = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.1)
    cbar_trajectory.set_label('Episode Progression')

    ticks = np.linspace(0, n-1, 10)
    cbar_trajectory.set_ticks(ticks)
    cbar_trajectory.set_ticklabels([f"{int(t)}" for t in ticks])

    #plt.show()
    save_plot_with_timestamp(fig, "all_trajectories")
    plt.close(fig)  # Close the figure to free memory
