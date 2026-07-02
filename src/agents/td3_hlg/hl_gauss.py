import jax
import jax.scipy.special
import jax.numpy as jnp
import flax.linen as nn
from flax.struct import PyTreeNode, field


DEFAULT_SIGMA_TO_BIN_RATIO = 0.75


class HLGauss(PyTreeNode):
    """Histogram loss transform for a normal distribution"""

    min_value: float
    max_value: float
    sigma: float
    support: jax.Array
    centres: jax.Array
    mean_bin_size: float
    clip_to_range: bool = field(pytree_node=False)

    @classmethod
    def create(
        cls,
        min_value,
        max_value,
        n_bins,
        sigma=None,
        sigma_to_bin_ratio=None,
        min_max_value_on_bin_centre=False,
        clip_to_range=True,
    ):
        if min_max_value_on_bin_centre:
            adjustment = (max_value - min_value) / ((n_bins - 1) * 2)
            min_value -= adjustment
            max_value += adjustment

        support = jnp.linspace(
            min_value, max_value, n_bins + 1, dtype=jnp.float32
        )
        centres = (support[:-1] + support[1:]) / 2

        # determine sigma
        mean_bin_size = (support[1:] - support[:-1]).mean().item()
        sigma_to_bin_ratio = (
            sigma_to_bin_ratio
            if sigma_to_bin_ratio is not None
            else DEFAULT_SIGMA_TO_BIN_RATIO
        )
        sigma = (
            sigma if sigma is not None else sigma_to_bin_ratio * mean_bin_size
        )

        return cls(
            min_value,
            max_value,
            sigma,
            support,
            centres,
            mean_bin_size,
            clip_to_range,
        )

    def transform_from_value_to_probs(
        self,
        target: jax.Array,
    ) -> jax.Array:
        if self.clip_to_range:
            target = target.clip(min=self.min_value, max=self.max_value)

        batch_size = target.shape[0]
        support = self.support[None, :].repeat(batch_size, axis=0)

        cdf_evals = jax.scipy.special.erf(
            (support - target[:, None]) / (jnp.sqrt(2) * self.sigma)
        )
        z = cdf_evals[:, -1] - cdf_evals[:, 0]

        bin_probs = cdf_evals[:, 1:] - cdf_evals[:, :-1]
        return bin_probs / z[:, None], z

    def transform_from_value_to_logprobs(
        self,
        probs: jax.Array,
        eps: float = 1e-20,
    ) -> jax.Array:
        probs = self.transform_from_value_to_probs(
            probs,
        )
        return jnp.log(probs.clip(min=eps))

    def transform_from_probs_to_value(
        self,
        probs: jax.Array,
    ) -> jax.Array:
        return (probs * self.centres).sum(-1)

    def transform_from_logits_to_value(
        self,
        logit: jax.Array,
    ) -> jax.Array:
        # for easier use
        prob = nn.softmax(logit, axis=-1)
        return self.transform_from_probs_to_value(
            prob,
        )
