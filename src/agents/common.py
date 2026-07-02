import jax
from flax.training.train_state import TrainState


def target_update(
    model: TrainState, target_model: TrainState, tau: float
) -> TrainState:
    new_target_params = jax.tree.map(
        lambda p, tp: p * tau + tp * (1 - tau),
        model.params,
        target_model.params,
    )

    return target_model.replace(params=new_target_params)
