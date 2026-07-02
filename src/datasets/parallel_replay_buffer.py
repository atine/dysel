import numpy as np
import gym

from src.common import Batch


class ParallelReplayBuffer:
    def __init__(
        self,
        state_dim: gym.spaces.Box,
        action_dim: int,
        capacity: int,
        num_seeds: int,
    ):
        self.observations = np.empty(
            (num_seeds, capacity, state_dim),
            dtype=np.float32,
        )
        self.actions = np.empty(
            (num_seeds, capacity, action_dim), dtype=np.float32
        )
        self.rewards = np.empty(
            (
                num_seeds,
                capacity,
            ),
            dtype=np.float32,
        )
        self.masks = np.empty(
            (
                num_seeds,
                capacity,
            ),
            dtype=np.float32,
        )
        self.dones_float = np.empty(
            (
                num_seeds,
                capacity,
            ),
            dtype=np.float32,
        )
        self.next_observations = np.empty(
            (num_seeds, capacity, state_dim),
            dtype=np.float32,
        )
        self.size = 0
        self.insert_index = 0
        self.capacity = capacity
        self.n_parts = 4

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        mask: float,
        done_float: float,
        next_observation: np.ndarray,
    ):
        self.observations[:, self.insert_index] = observation
        self.actions[:, self.insert_index] = action
        self.rewards[:, self.insert_index] = reward
        self.masks[:, self.insert_index] = mask
        self.dones_float[:, self.insert_index] = done_float
        self.next_observations[:, self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_parallel(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(
            observations=self.observations[:, indx],
            actions=self.actions[:, indx],
            rewards=self.rewards[:, indx],
            masks=self.masks[:, indx],
            next_observations=self.next_observations[:, indx],
        )

    def sample_parallel_multibatch(
        self, batch_size: int, num_batches: int
    ) -> Batch:
        indxs = np.random.randint(self.size, size=(num_batches, batch_size))
        return Batch(
            observations=self.observations[:, indxs],
            actions=self.actions[:, indxs],
            rewards=self.rewards[:, indxs],
            masks=self.masks[:, indxs],
            next_observations=self.next_observations[:, indxs],
        )
