from typing import Dict
import numpy as np


def evaluate(
    agent, env, num_episodes: int, episode_length: int
) -> Dict[str, float]:
    n_seeds = env.num_envs
    aggregate_dict = {"return": []}
    for _ in range(num_episodes):
        observations = env.reset()
        dones = np.array([False] * n_seeds)

        rets, length = np.zeros(n_seeds), 0
        while not dones.all():
            actions = agent.sample_actions(observations, temperature=0.0)
            prev_dones = dones
            observations, rewards, dones, infos = env.step(actions)

            rets += rewards * (1 - prev_dones)
            length += 1
            if length >= episode_length:
                break

        aggregate_dict["return"].append(rets)

    # calculate mean
    to_ret_dict = {}
    for k, v in aggregate_dict.items():
        to_ret_dict[k] = np.array(v).mean(axis=0)

    return to_ret_dict
