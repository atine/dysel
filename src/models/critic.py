from typing import Callable, Sequence, Tuple, Optional, Any
import jax
import jax.numpy as jnp
from flax import linen as nn
from src.models.common import MLP


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[Any], Any] = nn.relu
    layernorm: Optional[bool] = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(
        self,
        observations: jax.Array,
        actions: jax.Array,
        training: bool = True,
    ) -> jax.Array:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP(
            (*self.hidden_dims, 1),
            activations=self.activations,
            layernorm=self.layernorm,
            dropout_rate=self.dropout_rate,
        )(inputs, training=training)

        return jnp.squeeze(critic, -1)


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[Any], Any] = nn.relu
    layernorm: Optional[bool] = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(
        self,
        observations: jax.Array,
        actions: jax.Array,
        training: bool = True,
    ) -> Tuple[jax.Array, jax.Array]:
        critic1 = Critic(
            self.hidden_dims,
            activations=self.activations,
            layernorm=self.layernorm,
            dropout_rate=self.dropout_rate,
        )(observations, actions, training=training)
        critic2 = Critic(
            self.hidden_dims,
            activations=self.activations,
            layernorm=self.layernorm,
            dropout_rate=self.dropout_rate,
        )(observations, actions, training=training)

        return critic1, critic2


class DistributionalCritic(nn.Module):
    hidden_dims: Sequence[int]
    n_bins: int = 50
    activations: Callable[[Any], Any] = nn.relu
    layernorm: Optional[bool] = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(
        self,
        observations: jax.Array,
        actions: jax.Array,
        training: bool = True,
    ) -> jax.Array:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP(
            (*self.hidden_dims, self.n_bins),
            activations=self.activations,
            layernorm=self.layernorm,
            dropout_rate=self.dropout_rate,
        )(inputs, training=training)
        return critic


class DoubleDistributionalCritic(nn.Module):
    hidden_dims: Sequence[int]
    n_bins: int = 50
    activations: Callable[[Any], Any] = nn.relu
    layernorm: Optional[bool] = False

    @nn.compact
    def __call__(
        self,
        observations: jax.Array,
        actions: jax.Array,
        training: bool = True,
    ) -> Tuple[jax.Array, jax.Array]:
        critic1 = DistributionalCritic(
            self.hidden_dims,
            n_bins=self.n_bins,
            activations=self.activations,
            layernorm=self.layernorm,
        )(observations, actions, training=training)
        critic2 = DistributionalCritic(
            self.hidden_dims,
            n_bins=self.n_bins,
            activations=self.activations,
            layernorm=self.layernorm,
        )(observations, actions, training=training)
        return critic1, critic2
