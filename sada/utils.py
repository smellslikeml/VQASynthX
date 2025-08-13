# This is a simplified, representative implementation based on the source repo's structure.
import torch


class Cache_Bus:
    """A bus to hold state and configuration for a single generation pass."""

    def __init__(self, acc_range, **kwargs):
        self.acc_range = acc_range
        self.kwargs = kwargs
        self.reset()

    def reset(self):
        self.current_step = 0
        self.skipping_path = []

    def step(self):
        self.current_step += 1

    def is_pruning_step(self):
        # Determine if the current step is within the acceleration range
        return self.acc_range[0] <= self.current_step < self.acc_range[1]
