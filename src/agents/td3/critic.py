from typing import Tuple
import functools
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from src.common import InfoDict, Params, Batch


@functools.partial(jax.jit)
def update_critic(
    rng: jax.random.PRNGKey,
    target_actor: TrainState,
    critic: TrainState,
    target_critic: TrainState,
    batch: Batch,
    discount: float,
    policy_noise: float,
    policy_noise_clip: float,
    max_action: float,
) -> Tuple[TrainState, InfoDict]:
    rng, noise_key = jax.random.split(rng, 2)

    # action noise
    noise = (
        jax.random.normal(noise_key, shape=batch.actions.shape) * policy_noise
    )
    noise = jnp.clip(noise, -policy_noise_clip, policy_noise_clip)

    # next action with noise
    raw_next_actions = target_actor.apply_fn(
        target_actor.params, batch.next_observations
    )
    next_actions = jnp.clip(raw_next_actions + noise, -max_action, max_action)

    # next q
    next_q1, next_q2 = target_critic.apply_fn(
        target_critic.params,
        batch.next_observations,
        next_actions,
    )

    # next q, take minimum
    next_q = jnp.minimum(next_q1, next_q2)

    # target q
    target_q = batch.rewards + discount * batch.masks * next_q

    def critic_loss_fn(trainable_params: Params) -> Tuple[jax.Array, InfoDict]:
        q1, q2 = critic.apply_fn(
            trainable_params,
            batch.observations,
            batch.actions,
        )

        mse_loss1 = (q1 - target_q) ** 2
        mse_loss2 = (q2 - target_q) ** 2

        critic_loss = mse_loss1 + mse_loss2
        critic_loss = critic_loss.mean()
        info = {
            "critic_loss": critic_loss,
            "mse_loss1": mse_loss1.mean(),
            "mse_loss2": mse_loss2.mean(),
            "q1": q1.mean(),
            "q2": q2.mean(),
        }

        return critic_loss, info

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)

    return new_critic, info
