

def train_longer(env, episodes=5000, alpha=0.9, gamma=0.8, initial_Q=100):
    grid_shape = env.shape  # e.g., (N, N)
    n_states = grid_shape[0] * grid_shape[1]
    n_actions = len(actions)

    Q = np.full((n_states, n_actions), initial_Q, dtype=float)

    for ep in range(episodes):
        epsilon = 1 - ep / episodes
        state = env.reset()  # starting point: (x, y)
        state_idx = state[1] * grid_shape[0] + state[0]  # y * N + x

        done = False
        while not done:
            action = policy(Q, state_idx, epsilon)
            dx, dy = actions[action]
            next_state, velocity, done = env.step(state, (dx, dy))

            # Compute dt and reward
            v_norm = np.linalg.norm(velocity)
            a = env.grid_spacing  # whatever your lattice spacing is
            dt = a / v_norm if v_norm > 1e-6 else -1  # avoid division by zero

            if dt < 0:
                reward = -10
            elif done:
                reward = 10
            else:
                reward = -dt / 1000  # small cost for slow motion

            next_state_idx = next_state[1] * grid_shape[0] + next_state[0]

            # Q update
            best_next = np.max(Q[next_state_idx])
            Q[state_idx, action] += alpha * (reward + gamma * best_next - Q[state_idx, action])

            state = next_state
            state_idx = next_state_idx

    return Q


def train_template():
    """ Main training loop.
    """
    reward_list = []
    for episode in range(config['num_episodes']):
        state = environment.reset()
        done = False
        total_reward = 0

        for _ in range(config["num_steps"]):
            q_values = Q_Network(state)
            action = policy(q_values)
            next_state, reward, done = environment.step(state, action)
            Q_Network.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            reward_list.append(total_reward)
            if done: break

        print(f"Episode {episode}: Total Reward: {total_reward}")

def plot_all_trajectories2(trajectories, potential_fn=None):
    """
    Args:
        trajectories: list of trajectories, each trajectory is a list of (x, y)
        potential_fn: optional potential function to plot the background field
    """
    plt.figure(figsize=(8, 8))

    # Plot potential field
    if potential_fn:
        xx, yy = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        zz = potential_fn(xx, yy)
        plt.contourf(xx, yy, zz, levels=50, cmap="viridis", alpha=0.7)

    # Color map for episode progression
    cmap = plt.get_cmap("plasma")
    n = len(trajectories)

    for i in range(0, len(trajectories), 10): # old: for i, traj in enumerate(trajectories):
        traj = np.array(trajectories[i])
        x, y = traj[:, 0], traj[:, 1]
        color = cmap(i / (n - 1))
        plt.plot(x, y, color=color, linewidth=1.5, alpha=0.9)

    plt.title("Agent Trajectories Over Episodes")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), label='Episode Progression')
    plt.show()