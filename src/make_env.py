from typing import Optional
import gym

from dm_control import suite

from src.wrappers import SequentialMultiEnvWrapper


def get_script_implementation(env_name):
    """
    Factory function to load and return the correct script module.
    """
    # use gym version if it exists in gym; otherwise, use the dmc version
    env_ids_gym = [env_spec.id for env_spec in gym.envs.registry.all()]
    env_ids_dmc = ["-".join(k) for k in suite.ALL_TASKS]

    if env_name in env_ids_gym:
        from src.wrappers import (
            VideoRecorder,
            SequentialMultiEnvWrapper,
            DMCEnv,
            EpisodeMonitor,
            RepeatAction,
            SinglePrecision,
            StickyActionEnv,
        )

        return (
            "gym",
            VideoRecorder,
            SequentialMultiEnvWrapper,
            DMCEnv,
            EpisodeMonitor,
            RepeatAction,
            SinglePrecision,
            StickyActionEnv,
            gym.wrappers.FlattenObservation,
            gym.wrappers.RescaleAction,
        )

    elif env_name in env_ids_dmc:
        from src.wrappers import (
            VideoRecorder,
            SequentialMultiEnvWrapper,
            DMCEnv,
            EpisodeMonitor,
            RepeatAction,
            SinglePrecision,
            StickyActionEnv,
        )

        return (
            "dmc",
            VideoRecorder,
            SequentialMultiEnvWrapper,
            DMCEnv,
            EpisodeMonitor,
            RepeatAction,
            SinglePrecision,
            StickyActionEnv,
            gym.wrappers.FlattenObservation,
            gym.wrappers.RescaleAction,
        )

    else:
        raise NotImplementedError


def make_one_env(
    env_name: str,
    seed: int,
    save_folder: Optional[str] = None,
    add_episode_monitor: bool = True,
    action_repeat: int = 1,
    sticky: bool = False,
    flatten: bool = True,
):
    (
        env_type,
        VideoRecorder,
        SequentialMultiEnvWrapper,
        DMCEnv,
        EpisodeMonitor,
        RepeatAction,
        SinglePrecision,
        StickyActionEnv,
        FlattenObservation,
        RescaleAction,
    ) = get_script_implementation(env_name)

    if env_type == "gym":
        env = gym.make(env_name)

    elif env_type == "dmc":
        domain_name, task_name = env_name.split("-")
        env = DMCEnv(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs={"random": seed},
        )

    else:
        raise NotImplementedError

    if flatten and isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)

    env = RescaleAction(env, -1.0, 1.0)

    if add_episode_monitor:
        env = EpisodeMonitor(env)

    if action_repeat > 1:
        env = RepeatAction(env, action_repeat)

    if save_folder is not None:
        env = VideoRecorder(env, save_folder=save_folder)

    env = SinglePrecision(env)

    if sticky:
        env = StickyActionEnv(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env


def make_env(
    env_name: str,
    seed: int,
    save_folder: Optional[str] = None,
    add_episode_monitor: bool = True,
    action_repeat: int = 1,
    sticky: bool = False,
    flatten: bool = True,
    num_envs: Optional[int] = None,
):
    if num_envs is None:
        return make_one_env(
            env_name,
            seed,
            save_folder,
            add_episode_monitor,
            action_repeat,
            sticky,
            flatten,
        )
    else:
        env_fn_list = [
            lambda: make_one_env(
                env_name,
                seed + i,  # noqa
                save_folder,
                add_episode_monitor,
                action_repeat,
                sticky,
                flatten,
            )
            for i in range(num_envs)
        ]
        return SequentialMultiEnvWrapper(
            env_fn_list, [seed + i for i in range(num_envs)]
        )
