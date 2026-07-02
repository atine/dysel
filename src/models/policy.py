from typing import Optional, Sequence, Tuple, Callable
import functools
import flax.linen as nn
import jax
import numpy as np

from src.common import Params
from src.models.common import MLP, default_init


@functools.partial(jax.jit, static_argnames=("apply_fn", "distribution"))
@functools.partial(jax.vmap, in_axes=(0, None, 0, 0, None, None))
def _sample_actions(
    rng: jax.random.PRNGKey,
    apply_fn: Callable,
    actor_params: Params,
    observations: np.ndarray,
    temperature: float = 1.0,
    distribution: str = "log_prob",
) -> Tuple[jax.random.PRNGKey, jax.Array]:
    if distribution == "det":
        mean = apply_fn(actor_params, observations, temperature)
        return rng, mean
    else:
        dist = apply_fn(actor_params, observations, temperature)
        rng, key = jax.random.split(rng)
        return rng, dist.sample(seed=key)


def sample_actions(
    rng: jax.random.PRNGKey,
    apply_fn: Callable,
    actor_params: Params,
    observations: np.ndarray,
    temperature: float = 1.0,
    distribution: str = "log_prob",
) -> Tuple[jax.random.PRNGKey, jax.Array]:
    return _sample_actions(
        rng, apply_fn, actor_params, observations, temperature, distribution
    )


class MSEPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(
        self,
        observations: jax.Array,
        temperature: float = 1.0,
        training: bool = False,
    ) -> jax.Array:
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,
            dropout_rate=self.dropout_rate,
        )(observations, training=training)

        actions = nn.Dense(self.action_dim, kernel_init=default_init())(
            outputs
        )
        return nn.tanh(actions)
