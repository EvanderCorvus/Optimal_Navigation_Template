# Any utility functions that are used in multiple files can be placed here.
import numpy as np

def state_to_index(state, bounds, grid_size):
    x, y = state
    x_min, x_max, y_min, y_max = bounds
    x_idx = int((x - x_min) / (x_max - x_min) * grid_size)
    y_idx = int((y - y_min) / (y_max - y_min) * grid_size)
    x_idx = np.clip(x_idx, 0, grid_size - 1)
    y_idx = np.clip(y_idx, 0, grid_size - 1)
    return y_idx * grid_size + x_idx  # flatten