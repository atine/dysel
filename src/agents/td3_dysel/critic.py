from typing import Tuple
import functools
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from src.common import InfoDict, Params, Batch


DEFAULT_SIGMA_TO_BIN_RATIO = 0.75


def differentiable_erf_projection(
    target,
    support,
    bounds,
    target_sigma=0.3,
    sigma_to_bin_ratio=None,
    clip_to_range=True,
):
    if clip_to_range:
        target = target.clip(
            min=jnp.min(bounds, axis=-1),
            max=jnp.max(bounds, axis=-1),
        )

    sigma_to_bin_ratio = (
        sigma_to_bin_ratio
        if sigma_to_bin_ratio is not None
        else DEFAULT_SIGMA_TO_BIN_RATIO
    )

    widths = support[:, 1:] - support[:, :-1]
    mean_bin_size = widths.mean(1)
    sigma = sigma_to_bin_ratio * jax.lax.stop_gradient(mean_bin_size)
    sigma = jnp.maximum(target_sigma, sigma)

    cdf_evals = jax.scipy.special.erf(
        (support - target[:, None]) / (jnp.sqrt(2) * sigma[:, None])
    )

    z = cdf_evals[:, -1] - cdf_evals[:, 0]
    bin_probs = cdf_evals[:, 1:] - cdf_evals[:, :-1]
    probs = bin_probs / z[:, None]

    return probs, sigma, z


@functools.partial(jax.jit)
def update_critic_dysel(
    rng: jax.random.PRNGKey,
    target_actor: TrainState,
    critic: TrainState,
    target_critic: TrainState,
    adversary: TrainState,
    bounds: TrainState,
    target_bounds: TrainState,
    batch: Batch,
    discount: float,
    policy_noise: float,
    policy_noise_clip: float,
    max_action: float,
    target_bias: float,
    bounds_weight: float,
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
    _, next_q_centres, _ = target_bounds.apply_fn(
        target_bounds.params,
        batch.next_observations,
        next_actions,
    )
    next_q1 = (jax.nn.softmax(next_q1_logits, -1) * next_q_centres).sum(-1)
    next_q2 = (jax.nn.softmax(next_q2_logits, -1) * next_q_centres).sum(-1)

    # next q, take minimum
    next_q = jnp.minimum(next_q1, next_q2)

    # target q
    target_q = batch.rewards + discount * batch.masks * next_q

    # constraint alpha
    adv_alpha = adversary.apply_fn(adversary.params)
    # adv_alpha = jnp.clip(adv_alpha, max=1000)

    def critic_loss_fn(
        trainable_critic_params: Params, trainable_bounds_params: Params
    ) -> Tuple[jax.Array, InfoDict]:
        q1_logits, q2_logits = critic.apply_fn(
            trainable_critic_params,
            batch.observations,
            batch.actions,
        )
        q_supp, q_centres, q_bounds = bounds.apply_fn(
            trainable_bounds_params,
            batch.observations,
            batch.actions,
        )
        target_probs, sigma, target_z = differentiable_erf_projection(
            target_q, q_supp, q_bounds
        )

        # cross entropy
        ce_loss1 = optax.softmax_cross_entropy(q1_logits, target_probs)
        ce_loss2 = optax.softmax_cross_entropy(q2_logits, target_probs)
        ce_loss = ce_loss1 + ce_loss2

        # widths
        lower, upper = q_bounds[:, 0], q_bounds[:, 1]
        M = jnp.maximum(jnp.abs(lower), jnp.abs(upper))
        first_term = M * bounds_weight + ce_loss / bounds_weight

        # bias penalty
        projection_bias = 1.0 - target_z / 2
        violation = jax.nn.relu(projection_bias - target_bias)
        second_term = adv_alpha * violation

        # some additional calculations
        q1 = (jax.nn.softmax(q1_logits, -1) * q_centres).sum(-1)
        q2 = (jax.nn.softmax(q2_logits, -1) * q_centres).sum(-1)
        mse_loss1 = (q1 - target_q) ** 2
        mse_loss2 = (q2 - target_q) ** 2
        projected_mean = (target_probs * q_centres).sum(-1)
        value_bias = (projected_mean - target_q) ** 2

        # total
        critic_loss = first_term + second_term
        # critic_loss = first_term  # ablation 1: no mass
        # critic_loss = ce_loss + second_term  # ablation 2: no width
        critic_loss = critic_loss.mean()

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
            "bound1_upper": upper.mean(),
            "bound1_lower": lower.mean(),
            "sigma1": sigma.mean(),
            "first_term": first_term.mean(),
            "second_term": second_term.mean(),
        }

        return critic_loss, info

    (critic_grads, bounds_grads), critic_info = jax.grad(
        critic_loss_fn, has_aux=True, argnums=(0, 1)
    )(critic.params, bounds.params)
    new_critic = critic.apply_gradients(grads=critic_grads)
    new_bounds = bounds.apply_gradients(grads=bounds_grads)

    # new_bounds and new_target_z using new_bounds
    new_q_supp, new_q_centres, new_q_bounds = new_bounds.apply_fn(
        new_bounds.params,
        batch.observations,
        batch.actions,
    )
    _, _, new_target_z = differentiable_erf_projection(
        target_q, new_q_supp, new_q_bounds
    )

    # CARE: violation needs to be possible to be nagative, so no relu here
    new_projection_bias = 1.0 - new_target_z / 2
    violation = new_projection_bias - target_bias

    def adv_loss_fn(trainable_params):
        adv_alpha = adversary.apply_fn(trainable_params)
        loss = -(adv_alpha * violation).mean()
        info = {
            "loss_adv_alpha": loss,
            "violation": violation,
            "adv_alpha": adv_alpha,
        }

        return loss, info

    grads, adv_info = jax.grad(adv_loss_fn, has_aux=True)(adversary.params)
    new_adversary = adversary.apply_gradients(grads=grads)

    return new_critic, new_bounds, new_adversary, {**critic_info, **adv_info}
