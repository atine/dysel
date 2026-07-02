"""Implementations of algorithms for continuous control."""

from typing import Sequence, Tuple
import functools
import numpy as np
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from src.common import InfoDict, Batch
from src.models import (
    DoubleDistributionalCritic,
    MSEPolicy,
    sample_actions,
)
from src.agents.common import target_update
from .actor import update_actor
from .critic import update_critic_dysel
from .adversary import Adversary, Bounds


@dataclass(frozen=True)
class ConfigArgs:
    policy_update_freq: int
    max_steps: int
    max_action: float
    exploration_noise: float
    policy_noise: float
    policy_noise_clip: float
    init_min_value: float
    init_max_value: float
    n_bins: int
    target_bias: float
    bounds_weight: float


@functools.partial(
    jax.vmap, in_axes=(0, None, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None)
)
def _update(
    rng: jax.random.PRNGKey,
    step: int,
    actor: TrainState,
    target_actor: TrainState,
    critic: TrainState,
    target_critic: TrainState,
    adversary: TrainState,
    bounds: TrainState,
    target_bounds: TrainState,
    batch: Batch,
    discount: float,
    tau: float,
    args,
) -> Tuple[
    jax.random.PRNGKey,
    TrainState,
    TrainState,
    TrainState,
    TrainState,
    TrainState,
    InfoDict,
]:
    rng, critic_key = jax.random.split(rng, 2)

    new_critic, new_bounds, new_adversary, critic_info = update_critic_dysel(
        critic_key,
        target_actor,
        critic,
        target_critic,
        adversary,
        bounds,
        target_bounds,
        batch,
        discount,
        policy_noise=args.policy_noise,
        policy_noise_clip=args.policy_noise_clip,
        max_action=args.max_action,
        target_bias=args.target_bias,
        bounds_weight=args.bounds_weight,
    )

    def _apply_actor(_):
        new_actor, info = update_actor(
            actor,
            new_critic,
            target_bounds,
            batch,
        )
        return new_actor, info

    def _skip_actor(_):
        dummy_info = {
            "actor_loss": jnp.array(0.0),
        }
        return actor, dummy_info

    new_actor, actor_info = jax.lax.cond(
        step % args.policy_update_freq == 0,
        _apply_actor,
        _skip_actor,
        operand=None,
    )
    new_target_critic = target_update(new_critic, target_critic, tau)
    new_target_actor = target_update(new_actor, target_actor, tau)
    new_target_bounds = target_update(new_bounds, target_bounds, tau)

    info = {**critic_info, **actor_info}

    return (
        rng,
        new_actor,
        new_target_actor,
        new_critic,
        new_target_critic,
        new_adversary,
        new_bounds,
        new_target_bounds,
        info,
    )


@functools.partial(
    jax.jit, static_argnames=("discount", "tau", "num_updates", "args")
)
def _do_multiple_updates(
    rng: jax.random.PRNGKey,
    step: int,
    actor: TrainState,
    target_actor: TrainState,
    critic: TrainState,
    target_critic: TrainState,
    adversary: TrainState,
    bounds: TrainState,
    target_bounds: TrainState,
    batches: Batch,
    discount: float,
    tau: float,
    num_updates: int,
    args,
) -> Tuple[
    jax.random.PRNGKey,
    int,
    TrainState,
    TrainState,
    TrainState,
    TrainState,
    TrainState,
    InfoDict,
]:
    def one_step(i, state):
        (
            rng,
            step,
            actor,
            target_actor,
            critic,
            target_critic,
            adversary,
            bounds,
            target_bounds,
            info,
        ) = state
        (
            new_rng,
            new_actor,
            new_target_actor,
            new_critic,
            new_target_critic,
            new_adversary,
            new_bounds,
            new_target_bounds,
            info,
        ) = _update(
            rng,
            step,
            actor,
            target_actor,
            critic,
            target_critic,
            adversary,
            bounds,
            target_bounds,
            jax.tree.map(lambda x: jnp.take(x, i, axis=1), batches),
            discount,
            tau,
            args,
        )
        step = step + 1
        return (
            new_rng,
            step,
            new_actor,
            new_target_actor,
            new_critic,
            new_target_critic,
            new_adversary,
            new_bounds,
            new_target_bounds,
            info,
        )

    (
        rng,
        step,
        actor,
        target_actor,
        critic,
        target_critic,
        adversary,
        bounds,
        target_bounds,
        info,
    ) = one_step(
        0,
        (
            rng,
            step,
            actor,
            target_actor,
            critic,
            target_critic,
            adversary,
            bounds,
            target_bounds,
            {},
        ),
    )

    return jax.lax.fori_loop(
        1,
        num_updates,
        one_step,
        (
            rng,
            step,
            actor,
            target_actor,
            critic,
            target_critic,
            adversary,
            bounds,
            target_bounds,
            info,
        ),
    )


class TD3DyselLearner(object):
    def __init__(
        self,
        seed: int,
        # env settings
        state_dim: int,
        action_dim: int,
        # common RL settings
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        num_parallel_seeds: int = 1,
        max_steps: int = 3e6,
        max_action: float = 1.0,  # generally has no effect
        # td3
        td3_exploration_noise: float = 0.1,
        td3_policy_noise: float = 0.2,
        td3_policy_noise_clip: float = 0.5,
        # dysel
        init_min_value: float = -10.0,
        init_max_value: float = 10.0,
        n_bins: int = 128,
        bounds_weight: float = 1.0,
        # other tricks
        policy_update_freq: int = 2,
        num_updates: int = 1,
        layernorm: bool = False,
    ):
        self.seeds = jnp.arange(seed, seed + num_parallel_seeds)
        self.tau = tau
        self.discount = discount
        self.num_updates = num_updates

        self.args = ConfigArgs(
            policy_update_freq,
            max_steps,
            max_action,
            td3_exploration_noise,
            td3_policy_noise,
            td3_policy_noise_clip,
            init_min_value,
            init_max_value,
            n_bins,
            target_bias=0.005,
            bounds_weight=bounds_weight,
        )

        dummy_observations = jnp.ones([1, state_dim], dtype=jnp.float32)
        dummy_actions = jnp.ones([1, action_dim], dtype=jnp.float32)

        def _init_models(seed, n_bins):
            rng = jax.random.PRNGKey(seed)
            rng, actor_key, critic_key, adv_key, bounds_key = jax.random.split(
                rng, 5
            )

            # optimisers
            critic_optimiser = optax.adam(learning_rate=critic_lr)
            actor_optimiser = optax.adam(learning_rate=actor_lr)
            adv_optimiser = optax.adam(learning_rate=3e-04)
            bounds_optimiser = optax.adam(learning_rate=3e-04)

            # actors
            actor_def = MSEPolicy(hidden_dims, action_dim)
            actor = TrainState.create(
                apply_fn=actor_def.apply,
                params=actor_def.init(actor_key, dummy_observations),
                tx=actor_optimiser,
            )
            target_actor = TrainState.create(
                apply_fn=actor_def.apply,
                params=actor_def.init(actor_key, dummy_observations),
                tx=actor_optimiser,
            )

            # critics
            critic_def = DoubleDistributionalCritic(
                hidden_dims,
                n_bins=n_bins,
                layernorm=layernorm,
            )
            critic = TrainState.create(
                apply_fn=critic_def.apply,
                params=critic_def.init(
                    critic_key, dummy_observations, dummy_actions
                ),
                tx=critic_optimiser,
            )
            target_critic = TrainState.create(
                apply_fn=critic_def.apply,
                params=critic_def.init(
                    critic_key, dummy_observations, dummy_actions
                ),
                tx=critic_optimiser,
            )

            # adversary
            adversary_def = Adversary(init_adv_alpha=1.0)
            adversary = TrainState.create(
                apply_fn=adversary_def.apply,
                params=adversary_def.init(adv_key),
                tx=adv_optimiser,
            )

            # bounds
            bounds_def = Bounds(
                hidden_dims,
                n_bins=n_bins,
                init_min_value=self.args.init_min_value,
                init_max_value=self.args.init_max_value,
            )
            bounds = TrainState.create(
                apply_fn=bounds_def.apply,
                params=bounds_def.init(
                    bounds_key, dummy_observations, dummy_actions
                ),
                tx=bounds_optimiser,
            )
            target_bounds = TrainState.create(
                apply_fn=bounds_def.apply,
                params=bounds_def.init(
                    bounds_key, dummy_observations, dummy_actions
                ),
                tx=bounds_optimiser,
            )

            return (
                actor,
                target_actor,
                critic,
                target_critic,
                adversary,
                bounds,
                target_bounds,
                rng,
            )

        self.init_models = jax.jit(
            jax.vmap(_init_models, in_axes=(0, None)),
            static_argnames=["n_bins"],
        )

        (
            self.actor,
            self.target_actor,
            self.critic,
            self.target_critic,
            self.adversary,
            self.bounds,
            self.target_bounds,
            self.rng,
        ) = self.init_models(self.seeds, self.args.n_bins)
        self.trainable_models = ["actor", "critic", "adversary", "bounds"]
        self.step = 0

    def sample_actions(
        self, observations: np.ndarray, temperature: float = 1.0
    ) -> jax.Array:
        rng, actions = sample_actions(
            self.rng,
            self.actor.apply_fn,
            self.actor.params,
            observations,
            temperature,
            distribution="det",
        )
        self.rng = rng

        actions = np.asarray(actions)
        actions = (
            actions
            + np.random.normal(size=actions.shape)
            * self.args.exploration_noise
            * temperature
        )
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        (
            self.rng,
            self.step,
            self.actor,
            self.target_actor,
            self.critic,
            self.target_critic,
            self.adversary,
            self.bounds,
            self.target_bounds,
            info,
        ) = _do_multiple_updates(
            self.rng,
            self.step,
            self.actor,
            self.target_actor,
            self.critic,
            self.target_critic,
            self.adversary,
            self.bounds,
            self.target_bounds,
            batch,
            self.discount,
            self.tau,
            self.num_updates,
            self.args,
        )

        if self.step % 100 == 0:
            print(
                info["adv_alpha"].mean(),
                info["mass_z"].mean(),
                info["bound1_upper"].mean(),
                info["bound1_lower"].mean(),
                info["first_term"].mean(),
                info["second_term"].mean(),
            )

        return info

    def reset(self):
        (
            self.actor,
            self.target_actor,
            self.critic,
            self.target_critic,
            self.adversary,
            self.bounds,
            self.target_bounds,
            self.rng,
        ) = self.init_models(self.seeds, self.args.n_bins)
