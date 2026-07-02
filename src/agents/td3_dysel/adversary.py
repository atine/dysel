from typing import Sequence, Callable, Optional, Any
import jax
import jax.numpy as jnp
from flax import linen as nn

from src.models.common import MLP


DEFAULT_SIGMA_TO_BIN_RATIO = 0.75


class Bounds(nn.Module):
    hidden_dims: Sequence[int]
    n_bins: int = 128
    activations: Callable[[Any], Any] = nn.relu
    layernorm: Optional[bool] = False
    init_min_value: int = -10
    init_max_value: int = 10

    @nn.compact
    def __call__(
        self,
        observations: jax.Array,
        actions: jax.Array,
    ) -> jax.Array:
        inputs = jnp.concatenate([observations, actions], -1)
        x = MLP(
            self.hidden_dims,
            layernorm=self.layernorm,
            activations=self.activations,
            activate_final=True,
        )(inputs)
        bounds = nn.Dense(
            2,
            kernel_init=nn.initializers.uniform(scale=1e-4),
            bias_init=self.bounds_bias_init,
        )(x)

        # sort and calculate supports, centres
        a = jnp.min(bounds, axis=-1, keepdims=True)
        b = jnp.max(bounds, axis=-1, keepdims=True)
        steps = jnp.linspace(0, 1, self.n_bins + 1, dtype=jnp.float32)
        support = a + (b - a) * steps[None, :]
        centres = (support[..., :-1] + support[..., 1:]) / 2.0

        return support, centres, bounds

    def bounds_bias_init(self, key, shape, dtype=jnp.float32):
        return jnp.array(
            [self.init_min_value, self.init_max_value],
            dtype=dtype,
        )


class Adversary(nn.Module):
    init_adv_alpha: float = 1.0

    @nn.compact
    def __call__(self) -> jax.Array:
        log_adv_alpha = self.param(
            "log_adv_alpha",
            init_fn=lambda key: jnp.full((), jnp.log(self.init_adv_alpha)),
        )
        return jnp.exp(log_adv_alpha)
