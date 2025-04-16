import torch as tr
import torch.nn as nn
from torchvision.ops import MLP

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
        pass

    def update(self, state, action, reward, next_state, done):
        pass

# The Forward of the Network happens here (but you can change where it happens)
def policy(q_values):
    """ Maps q_values to Actions. Note that in principle a policy maps states to actions.
    For convenience, we use q_values.
    Args:
        q_values (np.ndarray): q values of each action.
    Returns:
        np.ndarray: Action to take.
    """
    pass