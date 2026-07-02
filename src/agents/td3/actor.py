from typing import Tuple
import functools
import jax
from flax.training.train_state import TrainState

from src.common import InfoDict, Params, Batch


@functools.partial(jax.jit)
def update_actor(
    actor: TrainState,
    critic: TrainState,
    batch: Batch,
) -> Tuple[TrainState, InfoDict]:
    def actor_loss_fn(trainable_params: Params) -> Tuple[jax.Array, InfoDict]:
        pi_actions = actor.apply_fn(trainable_params, batch.observations)
        q, _ = critic.apply_fn(critic.params, batch.observations, pi_actions)

        actor_loss = -q.mean()
        info = {
            "actor_loss": actor_loss,
        }

        return actor_loss, info

    grads, info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info
