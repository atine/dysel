# common
import os
import copy
import random
import hydra
import functools
import numpy as np
import wandb
import socket
from tqdm import tqdm as std_tqdm
from omegaconf import OmegaConf
from datetime import datetime

# deep RL related
import jax
import jax.numpy as jnp
import flax


from src.datasets import ParallelReplayBuffer
from src.evaluation import evaluate
from src.make_env import make_env


tqdm = functools.partial(std_tqdm, dynamic_ncols=True)


def make_agent_and_update_configs(seed, state_dim, action_dim, cfg):
    """!! task settings supercedes algo settings !!"""

    # backup
    old_cfg = copy.deepcopy(cfg)

    # update tags if utd is in additional
    if "utd" in cfg.additional or cfg.agent_updates_per_step != 1:
        tags = cfg.algo.algo_tags
        tags = tags + ["utd"]
        OmegaConf.update(cfg.algo, "algo_tags", tags)

    # FOR TESTING/DEBUGGING: reset some configs if testing
    if cfg.supercedes_all_settings:
        cfg.eval_episodes = old_cfg.eval_episodes
        cfg.eval_freq = old_cfg.eval_freq
        cfg.max_timesteps = old_cfg.max_timesteps
        cfg.show_stats_freq = old_cfg.show_stats_freq
        cfg.no_track = old_cfg.no_track
        cfg.seed = old_cfg.seed

        if hasattr(cfg.task, "eval_freq"):
            cfg.task.eval_freq = cfg.eval_freq
        if hasattr(cfg.task, "eval_episodes"):
            cfg.task.eval_episodes = cfg.eval_episodes

    # finally update configs into algo configs,
    # because task configs are not stored
    algo_name_orig = cfg.algo.algo_name
    if cfg.additional is not None and cfg.additional != "":
        OmegaConf.update(
            cfg.algo,
            "algo_name",
            "".join([cfg.algo.algo_name, cfg.additional]),
            merge=False,
        )
    OmegaConf.update(cfg.algo.agent, "seed", seed, merge=False)
    OmegaConf.update(cfg.algo.agent, "state_dim", state_dim, merge=False)
    OmegaConf.update(cfg.algo.agent, "action_dim", action_dim, merge=False)

    # update task-specific configs for algos: use base algo configs
    # for example, algos derived from td3 will use td3 configs
    # for example, td3_ivon will use td3 configs
    algo_name_orig = algo_name_orig.split("_")[0]
    if hasattr(cfg.task, algo_name_orig):
        for additional_config in getattr(cfg.task, algo_name_orig):
            OmegaConf.update(
                cfg.algo.agent,
                additional_config,
                getattr(getattr(cfg.task, algo_name_orig), additional_config),
                merge=False,
                force_add=True,
            )

    return hydra.utils.instantiate(cfg.algo.agent)


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f"workspace: {self.work_dir}")
        self.cfg = cfg
        self.cfg.additional = str(self.cfg.additional)

        # setup env
        self.train_env = make_env(
            self.cfg.task.task_name,
            self.cfg.seed,
            num_envs=self.cfg.num_parallel_seeds,
        )
        self.eval_env = make_env(
            self.cfg.task.task_name,
            self.cfg.seed + 100,
            num_envs=self.cfg.num_parallel_seeds,
        )
        self.train_env.reset()
        self.eval_env.reset()

        # setup agent
        state_dim = self.train_env.observation_space.shape[1]
        action_dim = self.train_env.action_space.shape[1]
        np.random.seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        self.agent = make_agent_and_update_configs(
            cfg.seed, state_dim, action_dim, self.cfg
        )

        # setup replay buffer
        self.replay_buffer = ParallelReplayBuffer(
            state_dim,
            action_dim,
            int(1e6),
            num_seeds=self.cfg.num_parallel_seeds,
        )

    def looper(self):
        observations = self.train_env.reset()
        dones = False
        rewards = 0.0
        infos = {}
        eval_returns = [[] for _ in range(self.cfg.num_parallel_seeds)]
        max_timesteps = int(self.cfg.max_timesteps)
        pbar = tqdm(range(max_timesteps))

        # main timesteps loop
        for curr_timestep in pbar:
            if curr_timestep < self.cfg.init_collection_steps:
                actions = self.train_env.action_space.sample()
            else:
                actions = self.agent.sample_actions(observations)

            # env step
            next_observations, rewards, dones, infos = self.train_env.step(
                actions
            )
            masks = self.train_env.generate_masks(dones, infos)

            # save to replay buffer
            self.replay_buffer.insert(
                observations, actions, rewards, masks, dones, next_observations
            )
            observations, dones = self.train_env.reset_where_done(
                next_observations, dones
            )

            # updates
            if curr_timestep >= self.cfg.init_collection_steps:
                batches = self.replay_buffer.sample_parallel_multibatch(
                    self.cfg.batch_size, self.cfg.agent_updates_per_step
                )
                infos = self.agent.update(batches)

                # log to wandb if time to
                if curr_timestep % self.cfg.show_stats_freq == 0:
                    self.log_multiple_seeds_to_wandb(curr_timestep, infos)

            # if eval
            self.evaluate_if_time_to(
                curr_timestep,
                eval_returns,
                infos,
                list(
                    range(
                        self.cfg.seed,
                        self.cfg.seed + self.cfg.num_parallel_seeds,
                    )
                ),
            )

            # if reset
            self.reset_if_time_to(curr_timestep)

        # last eval
        self.evaluate_if_time_to(
            curr_timestep + 1,
            eval_returns,
            infos,
            list(
                range(
                    self.cfg.seed,
                    self.cfg.seed + self.cfg.num_parallel_seeds,
                )
            ),
        )

    def evaluate_if_time_to(self, curr_timestep, eval_returns, info, seeds):
        # if eval
        if curr_timestep % self.cfg.eval_freq == 0:
            eval_stats = evaluate(
                self.agent,
                self.eval_env,
                self.cfg.eval_episodes,
                episode_length=1000,
            )

            if not self.cfg.no_disc_logging:
                for j, seed in enumerate(seeds):
                    eval_returns[j].append(
                        (curr_timestep, eval_stats["return"][j])
                    )
                    np.savetxt(
                        os.path.join(self.work_dir, f"{seed}.txt"),
                        eval_returns[j],
                        fmt=["%d", "%.1f"],
                    )
            self.log_multiple_seeds_to_wandb(curr_timestep, eval_stats)

    def reset_if_time_to(self, curr_timestep):
        # if reset
        if self.cfg.resets and curr_timestep % self.cfg.resets_freq == 0:
            self.agent.reset()

    def log_multiple_seeds_to_wandb(self, step, infos):
        dict_to_log = {}
        for info_key in infos:
            for seed, value in enumerate(infos[info_key]):
                dict_to_log[f"seed{seed}/{info_key}"] = value
        if not self.cfg.no_track:
            dict_to_log["step"] = step
            wandb.log(dict_to_log, step=step)


@hydra.main(
    version_base=None, config_path="cfgs", config_name="default_config"
)
def main(cfg):
    print("=" * 65)
    print()
    start_time = datetime.now()
    start_time = start_time.strftime("%Y/%m/%d %H:%M:%S")
    print(f"start time: {start_time}")

    # set workspace
    root_dir = os.getcwd()
    workspace = Workspace(cfg)

    # run with wandb logging
    if not cfg.no_track:
        run = wandb.init(
            project="flexible-dysel",
            config=OmegaConf.to_container(cfg, resolve=True),
            name=cfg.algo.algo_name,
            tags=cfg.algo.algo_tags,
        )
        algo_root = os.path.abspath(
            os.path.dirname(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    os.path.dirname(cfg.algo.agent._target_.replace(".", "/")),
                )
            )
        )
        run.log_code(root=algo_root)

    # print model definitions
    print("-" * 40)
    print()
    print("model definitions:")
    for trm in workspace.agent.trainable_models:
        flat_params = flax.traverse_util.flatten_dict(
            getattr(workspace.agent, trm).params, sep="/"
        )
        flat_params_dict = jax.tree_util.tree_map(jnp.shape, flat_params)
        print()
        print(trm)
        for k, v in flat_params_dict.items():
            print(k, v)
        if not workspace.cfg.no_disc_logging:
            with open(f"{workspace.work_dir}/{trm}_definition.txt", "w") as f:
                for k, v in flat_params_dict.items():
                    f.write("%s:%s\n" % (k, v))
    print()

    # print configs
    print("configs:")
    configyaml = OmegaConf.to_yaml(workspace.cfg, resolve=True)
    print(configyaml)

    if not workspace.cfg.no_disc_logging:
        OmegaConf.save(
            workspace.cfg, f"{workspace.work_dir}/config.yaml", resolve=True
        )
    print()

    # looper
    workspace.looper()

    # finish
    if not cfg.no_track:
        wandb.alert(
            title=cfg.algo.algo_name,
            text=(
                f"Hostname: {socket.gethostname()}, "
                f"Task: {cfg.task.read_task_name}"
            ),
        )
        wandb.finish()

    # print train time
    end_time = datetime.now()
    end_time = end_time.strftime("%Y/%m/%d %H:%M:%S")
    print(f"workspace: {root_dir}")
    print(f"start time: {start_time}")
    print(f"end time: {end_time}")


if __name__ == "__main__":
    main()
