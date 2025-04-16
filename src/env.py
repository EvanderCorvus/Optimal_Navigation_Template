# Optional: Use Gymnasium Library for environment creation
import numpy as np

class Env:
    def __init__(self, config):
        pass

    def reset(self) -> np.ndarray:
        """ Reset environment.
        Returns:
            np.ndarray: Initial state of the environment.
        """
        pass

    def step(self, state, action) -> tuple[np.ndarray, float, bool]:
        """ Perform a Langevin Time Dynamics Step.
        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Action to take.
        Returns:
            tuple[np.ndarray, float, bool]: Next state, Reward, Done flag.
        """
        pass
    
    def _get_reward(self, state, action) -> float:
        """ Calculate reward based on state and action.
        Args:
            state (np.ndarray): Relevant State for the reward.
            action (np.ndarray): Action taken.
        Returns:
            float: Reward.
        """
        pass
