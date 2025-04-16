from agent import DeepQNetwork, policy
from env import Env
from utils import *
import json
import torch as tr

with open('hyperparams.json') as f:
    config = json.load(f)
device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')

environment = Env(config)
Q_Network = DeepQNetwork(config).to(device)

def train():
    """ Main training loop.
    """
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
            if done: break

        print(f"Episode {episode}: Total Reward: {total_reward}")

if __name__ == "__main__":
    train()
    