import jax

# from jax.lib import xla_bridge
# from jax import extend
import jax.numpy as jnp
import timeit


# print(jax.extend.backend.get_backend())
print(jax.default_backend())
print(jax.random.normal(jax.random.key(0), shape=(8, 1)))


def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)


loop = 1000
key = jax.random.key(1701)
x = jax.random.normal(key, (1_000_000,))
result = timeit.timeit(
    "selu(x).block_until_ready()", globals=globals(), number=loop
)
print(result)
