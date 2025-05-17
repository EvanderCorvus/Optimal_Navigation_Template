# Optional: Use Gymnasium Library for environment creation
import numpy as np

class Env:
    def __init__(self, config):
        self.config = config
        self.dt = config.get("dt", 0.01)
        self.U0 = config.get("U0", 0.25)
        self.vortex_strength = config.get("vortex_strength", 0.1)
        self.reset()

    def reset(self):
        """ Reset environment.
        Returns:
            np.ndarray: Initial state of the environment.
        """
        self.position = np.array([-0.5, 0.0])  # Start at left
        return self.position.copy()

    def step(self, state, action):
        """ Perform a Langevin Time Dynamics Step.
        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Action to take.
        Returns:
            tuple[np.ndarray, float, bool]: Next state, Reward, Done flag.
        """    
        e = action / np.linalg.norm(action)  # normalize the orientation vector
        F = -self._grad_potential(state)
        v = e + F  # total velocity
        new_state = state + self.dt * v

        # Compute reward and check if done
        reward = self._get_reward(new_state, action)
        done = np.linalg.norm(new_state - np.array([0.5, 0.0])) < 0.05  # goal at x=0.5
        return new_state, reward, done

    def _get_reward(self, state, action):
        """ Calculate reward based on state and action.
        Args:
            state (np.ndarray): Relevant State for the reward.
            action (np.ndarray): Action taken.
        Returns:
            float: Reward.
        """
        # Negative reward to encourage reaching the goal quickly and penalty for large actions
        distance_to_goal = np.linalg.norm(state - np.array([0.5, 0.0]))
        penalty = 0.1 * np.linalg.norm(action)
        #direction_change_penalty = np.linalg.norm(action - prev_action) #needs to be defined
        # Reward shaping
        if distance_to_goal < 0.05:
            return 10.0  # Strong positive reward
        else:
            return -distance_to_goal  - penalty #- 0.1 * direction_change_penalty

    def _potential(self, pos):
        x, y = pos
        rho = np.sqrt(x**2 + y**2)

        # Mexican hat
        U_hat = 16 * self.U0 * (rho**2 - 0.25)**2 if rho <= 0.5 else 0

        # Vortex component
        U_vortex = self.vortex_strength * np.arctan2(-x, y) # to avoid discontinuity on the left where the agent starts

        return U_hat + U_vortex

    def _grad_potential(self, pos, eps=1e-5):
        x, y = pos
        dU_dx = (self._potential([x + eps, y]) - self._potential([x - eps, y])) / (2 * eps)
        dU_dy = (self._potential([x, y + eps]) - self._potential([x, y - eps])) / (2 * eps)
        return np.array([dU_dx, dU_dy])

