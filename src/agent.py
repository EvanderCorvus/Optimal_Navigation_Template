import torch as tr
import torch.nn as nn
from torchvision.ops import MLP
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = MLP(
            config['state_dim']+config['action_dim'],
            config['hidden_dims'],
            activation_layer = nn.LeakyReLU,
        )
        self.output_layer = nn.Linear(
            config['hidden_dims'][-1],
            config['action_dim'],
        )

        self.optimizer = tr.optim.Adam(
            self.parameters(),
            lr=config['learning_rate'],
        )
        self.loss = nn.MSELoss()
    
    def forward(self, state, action):
        """
        Args:
            state: tensor of shape [batch_size, state_dim]
            action: tensor of shape [batch_size, action_dim]
        Returns:
            Q-values: tensor of shape [batch_size, 1]
        """
        x = torch.cat([state, action], dim=-1)  # concat along features
        x = self.net(x)
        q_value = self.output_layer(x)
        return q_value


    def update(self, state, action, reward, next_state, done, gamma=0.99):
        """
        Perform one step of Q-learning update.
        Args:
            state, action, next_state: torch tensors of shape [batch_size, ...]
            reward, done: torch tensors of shape [batch_size, 1]
        """
        self.optimizer.zero_grad()

        # Predict Q(s, a)
        q_pred = self.forward(state, action)

        # Compute max_a' Q(s', a') for next actions
        with torch.no_grad():
            next_actions = self.sample_actions(next_state)  # define this separately
            q_next = self.forward(next_state, next_actions)
            q_target = reward + (1 - done.float()) * gamma * q_next

        loss = self.loss(q_pred, q_target)
        loss.backward()
        self.optimizer.step()

    def sample_actions(self, state, n_samples=10):
        """For each state, sample actions and pick the one with highest Q"""
        batch_size = state.shape[0]
        best_actions = []

        for i in range(batch_size):
            s = state[i].unsqueeze(0).repeat(n_samples, 1)
            a = torch.randn(n_samples, self.config['action_dim'])  # or use your custom sampling
            q_vals = self.forward(s, a).squeeze()
            best_idx = torch.argmax(q_vals)
            best_actions.append(a[best_idx])

        return torch.stack(best_actions)



# The Forward of the Network happens here (but you can change where it happens)
def policy(q_values, epsilon):
    """ Maps Q to Actions. Note that in principle a policy maps states to actions.
    For convenience, we use Q.
    Args:
        Q (np.ndarray): q values of each action.
    Returns:
        np.ndarray: Action to take.
    """
    if np.random.rand() < epsilon:
        return np.random.randint(len(q_values))  # Explore
    else:
        return np.argmax(q_values)  # Exploit