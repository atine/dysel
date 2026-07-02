from typing import Tuple
import functools
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.struct import PyTreeNode

from src.common import InfoDict, Params, Batch


@functools.partial(jax.jit)
def update_critic_hlg(
    rng: jax.random.PRNGKey,
    target_actor: TrainState,
    critic: TrainState,
    target_critic: TrainState,
    batch: Batch,
    discount: float,
    policy_noise: float,
    policy_noise_clip: float,
    max_action: float,
    # hlg
    hl_gauss: PyTreeNode,
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
    next_q1_logits, next_q2_logits = target_critic.apply_fn(
        target_critic.params,
        batch.next_observations,
        next_actions,
    )
    next_q1 = hl_gauss.transform_from_logits_to_value(next_q1_logits)
    next_q2 = hl_gauss.transform_from_logits_to_value(next_q2_logits)

    # next q, take minimum
    next_q = jnp.minimum(next_q1, next_q2)

    # target q
    target_q = batch.rewards + discount * batch.masks * next_q
    target_probs, target_z = hl_gauss.transform_from_value_to_probs(target_q)

    def critic_loss_fn(trainable_params: Params) -> Tuple[jax.Array, InfoDict]:
        q1_logits, q2_logits = critic.apply_fn(
            trainable_params,
            batch.observations,
            batch.actions,
        )

        # cross entropy
        ce_loss1 = optax.softmax_cross_entropy(q1_logits, target_probs)
        ce_loss2 = optax.softmax_cross_entropy(q2_logits, target_probs)
        ce_loss = ce_loss1 + ce_loss2

        # some additional calculations
        q1 = hl_gauss.transform_from_logits_to_value(q1_logits)
        q2 = hl_gauss.transform_from_logits_to_value(q2_logits)
        mse_loss1 = (q1 - target_q) ** 2
        mse_loss2 = (q2 - target_q) ** 2
        projected_mean = hl_gauss.transform_from_probs_to_value(target_probs)
        value_bias = (projected_mean - target_q) ** 2

        # total
        critic_loss = ce_loss.mean()

        info = {
            "critic_loss": critic_loss,
            "ce_loss1": ce_loss1.mean(),
            "ce_loss2": ce_loss2.mean(),
            "mse_loss1": mse_loss1.mean(),
            "mse_loss2": mse_loss2.mean(),
            "q1": q1.mean(),
            "q2": q2.mean(),
            "q1_logits": q1_logits.mean(),
            "q2_logits": q2_logits.mean(),
            "q1_prob": jax.nn.softmax(q1_logits, axis=-1).mean(),
            "q2_prob": jax.nn.softmax(q2_logits, axis=-1).mean(),
            "target_prob": target_probs.sum(-1).mean(),
            "bias": value_bias.mean(),
            "mass_z": target_z.mean(),
            # additional loggings
            "bound1_upper": hl_gauss.max_value.mean(),
            "bound1_lower": hl_gauss.min_value.mean(),
            "sigma1": hl_gauss.sigma.mean(),
            "bin_size": hl_gauss.mean_bin_size.mean(),
        }

        return critic_loss, info

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)

    return new_critic, info
