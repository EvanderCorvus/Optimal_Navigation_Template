from agent import DeepQNetwork, policy
from env import Env
from utils import *
from plotting import *
import json
import torch as tr
import numpy as np

with open('hyperparams.json') as f:
    config = json.load(f)
device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')

environment = Env(config)
Q_Network = DeepQNetwork(config).to(device)

actions = {
    0: np.array([1, 0]),    # right
    1: np.array([0, 1]),    # up
    2: np.array([-1, 0]),   # left
    3: np.array([0, -1]),   # down
    4: np.array([1, 1]) / np.sqrt(2),   # up-right
    5: np.array([-1, 1]) / np.sqrt(2),  # up-left
    6: np.array([-1, -1]) / np.sqrt(2), # down-left
    7: np.array([1, -1]) / np.sqrt(2),  # down-right
}

class QTable:
    def __init__(self, n_states, n_actions, init_val=0.0):
        self.Q = np.full((n_states, n_actions), init_val, dtype=float)

    def __call__(self, state_idx):
        return self.Q[state_idx]

    def update(self, state_idx, action_idx, reward, next_state_idx, done, alpha, gamma):
        max_q_next = 0 if done else np.max(self.Q[next_state_idx])
        td_target = reward + gamma * max_q_next
        td_error = td_target - self.Q[state_idx, action_idx]
        self.Q[state_idx, action_idx] += alpha * td_error

reward_list = []
all_trajectories = []

def train():
    bounds = (-1.0, 1.0, -1.0, 1.0)  # assuming your env operates in this region
    grid_size = 50  # 20x20 state grid
    n_states = grid_size * grid_size
    n_actions = len(actions)

    Q_Network = QTable(n_states, n_actions, init_val=0.0)
    epsilon = config.get("epsilon", 1.0)
    decay_rate = config.get("epsilon_decay", 0.99)  # slow decay
    alpha = config.get("alpha", 0.1)
    gamma = config.get("gamma", 0.99)

    for episode in range(config['num_episodes']):
        state = environment.reset()
        episode_states = []  # for trajectory plotting
        done = False
        total_reward = 0

        for _ in range(config["num_steps"]):
            s_idx = state_to_index(state, bounds, grid_size)
            q_vals = Q_Network(s_idx)
            a_idx = policy(q_vals, epsilon)
            action = actions[a_idx]

            next_state, reward, done = environment.step(state, action)
            episode_states.append(next_state)
            ns_idx = state_to_index(next_state, bounds, grid_size)

            Q_Network.update(s_idx, a_idx, reward, ns_idx, done, alpha, gamma)

            state = next_state
            total_reward += reward
            if done: break

        reward_list.append(total_reward)
        all_trajectories.append(episode_states)
        epsilon *= decay_rate # Decay epsilon
        print(f"Episode {episode}: Total Reward: {total_reward}")


if __name__ == "__main__":
    train()
    plot_rewards(reward_list)
    plot_all_trajectories(all_trajectories, potential_fn=potential_fn)
