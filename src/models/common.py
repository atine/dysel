from typing import Callable, Optional, Sequence, Any, Dict, Type

import flax.linen as nn
import jax
import jax.numpy as jnp


_ACTIVATION_CACHE: Dict[str, Type[nn.Module]] = {}


def default_init(scale: Optional[float] = jnp.sqrt(2)):  # noqa
    return nn.initializers.orthogonal(scale)


def get_activation(fn: Callable) -> nn.Module:
    name = fn.__name__.capitalize()
    if name not in _ACTIVATION_CACHE:

        class WrappedActivation(nn.Module):
            @nn.compact
            def __call__(self, x):
                self.variable('constants', 'activations', lambda: jnp.array(1.0))
                return fn(x)

        WrappedActivation.__name__ = name
        WrappedActivation.__qualname__ = name
        _ACTIVATION_CACHE[name] = WrappedActivation

    return _ACTIVATION_CACHE[name]()


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[Any], Any] = nn.relu
    activate_final: Optional[bool] = False
    layernorm: Optional[bool] = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )

                if self.layernorm:
                    x = nn.LayerNorm()(x)

                x = get_activation(self.activations)(x)

        return x
