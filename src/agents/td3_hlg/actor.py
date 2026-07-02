from typing import Tuple
import functools
import jax
from flax.training.train_state import TrainState
from flax.struct import PyTreeNode

from src.common import InfoDict, Params, Batch


@functools.partial(jax.jit)
def update_actor(
    actor: TrainState,
    critic: TrainState,
    batch: Batch,
    # hlg
    hl_gauss: PyTreeNode,
) -> Tuple[TrainState, InfoDict]:
    def actor_loss_fn(trainable_params: Params) -> Tuple[jax.Array, InfoDict]:
        pi_actions = actor.apply_fn(trainable_params, batch.observations)
        q_logits, _ = critic.apply_fn(
            critic.params, batch.observations, pi_actions
        )
        q = hl_gauss.transform_from_logits_to_value(q_logits)

        actor_loss = -q.mean()
        info = {
            "actor_loss": actor_loss,
        }

        return actor_loss, info

    grads, info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info
