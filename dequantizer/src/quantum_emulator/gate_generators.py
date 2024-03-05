import jax.numpy as jnp
from jax import Array
from jax.random import normal


def _get_random_ising_like_q2_gate(key: Array) -> Array:
    diag = normal(key, (4,))
    return jnp.diag(jnp.exp(1j * diag)).reshape((2, 2, 2, 2))


def _get_random_q1_gate(key: Array) -> Array:
    gate = normal(key, (2, 2, 2))
    gate = gate[..., 0] + 1j * gate[..., 1]
    gate, _ = jnp.linalg.qr(gate)
    return gate
