import subprocess
import os
import pathlib
import yaml
import importlib
import collections
import natsort
from functools import partial
from tqdm import tqdm as std_tqdm
from io import BytesIO
import pandas as pd
import warnings


def get_free_gpu():
    gpu_stats = subprocess.check_output(
        ["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]
    )
    gpu_df = pd.read_csv(
        BytesIO(gpu_stats), names=["memory.used", "memory.free"], skiprows=1
    )
    print("GPU usage:\n{}".format(gpu_df))
    gpu_df["memory.free"] = gpu_df["memory.free"].map(
        lambda x: x.rstrip(" [MiB]")
    )
    idx = gpu_df["memory.free"].idxmax()
    print(
        "Returning GPU{} with {} free MiB".format(
            idx, gpu_df.iloc[idx]["memory.free"]
        )
    )
    return idx


warnings.filterwarnings("ignore")
free_gpu_id = get_free_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpu_id)
print(f"setting to gpu {free_gpu_id}")


import jax  # noqa
import jax.numpy as jnp  # noqa


# configs
Batch = collections.namedtuple(
    "Batch",
    [
        "observations",
        "actions",
        "rewards",
        "masks",
        "next_observations",
    ],
)
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
tqdm = partial(std_tqdm, dynamic_ncols=True)


seed = 0
num_seeds = 10
state_dim = 17
action_dim = 6
batch_size = 4
dummy_obs = jnp.ones([num_seeds, 1, batch_size, state_dim])
dummy_actions = jnp.ones([num_seeds, 1, batch_size, action_dim])
dummy_rewards = jnp.ones([num_seeds, 1, batch_size])
dummy_masks = jnp.ones([num_seeds, 1, batch_size])
dummy_next_obs = jnp.ones([num_seeds, 1, batch_size, state_dim]) * 0.5
batch = Batch(
    dummy_obs,
    dummy_actions,
    dummy_rewards,
    dummy_masks,
    dummy_next_obs,
)


def initialise_agent(agent_target_string):
    # agent_target_string is something like
    # 'src.agents.td3.td3_learner.TD3Learner'

    fp, agentname = agent_target_string.rsplit(".", 1)
    module = importlib.import_module(fp)
    agent = getattr(module, agentname)(
        seed, state_dim, action_dim, num_parallel_seeds=num_seeds
    )
    return agent


def main():
    # get agent names
    print()
    print("=" * 65)
    print("start\n")

    agent_names = [p.path for p in os.scandir("cfgs/algo/")]
    agent_names = natsort.natsorted(
        [
            agent_name
            for agent_name in agent_names
            if pathlib.Path(agent_name).suffix == ".yaml"
        ]
    )
    print(f"total {len(agent_names)} agents:")
    for agent_name in agent_names:
        print(agent_name)

    # get agent class name from config yamls
    agent_target_strings = []
    for agent in agent_names:
        with open(agent, "r") as stream:
            data_loaded = yaml.safe_load(stream)
            agent_target_strings.append(data_loaded["agent"]["_target_"])

    # initialise agents
    print()
    print("=" * 65)
    print("initialise\n")

    agents = []
    pbar = tqdm(agent_target_strings, total=len(agent_target_strings))
    for agent_target_string in pbar:
        pbar.set_description(f"initialising {agent_target_string}")
        agents.append(initialise_agent(agent_target_string))
    pbar.close()

    # test all agents for one dummy batch
    print()
    print("=" * 65)
    print("test\n")

    pbar = tqdm(zip(agents, agent_names), total=len(agents))
    for agent, agent_name in pbar:
        pbar.set_description(f"testing {agent_name}")
        _ = agent.update(batch)
    pbar.close()
    print()


if __name__ == "__main__":
    main()
