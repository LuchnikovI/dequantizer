from typing import Callable, List
import jax.numpy as jnp
from jax.random import normal
from jax import Array
from .node import Node


def _gen_ghz_core(phys_dimension: int, modes_number: int) -> Array:
    if phys_dimension == 1:
        return jnp.array(1.0, dtype=jnp.complex64).reshape(modes_number * (1,))
    stride = (phys_dimension**modes_number - 1) // (phys_dimension - 1)
    elements_number = phys_dimension**modes_number
    core = jnp.zeros(elements_number, dtype=jnp.complex64)
    core = core.at[0:elements_number:stride].set(1.0)
    return core.reshape(modes_number * (phys_dimension,))


"""Returns an initializer that initializes a tensor from i.i.d. complex normal distribution with 0 mean and std 1.
Args:
    node: node of a tensor graph.
Returns:
    Initializer."""


def get_tensor_random_normal_initializer(key: Array) -> Callable[[Node], Array]:
    def initializer(node: Node) -> Array:
        shape = (*node.bond_shape, node.dimension, 2)
        tensor = normal(key, shape)
        tensor = tensor[..., 0] + 1j * tensor[..., 1]
        return tensor

    return initializer


"""Returns an initializer that initializes a tensor in such  way that the resulting tensor graph state is
|0>.
Args:
    node: node of a tensor graph.
Returns:
    Initializer."""


def get_tensor_std_state_initializer() -> Callable[[Node], Array]:
    def initializer(node: Node) -> Array:
        tensor = jnp.zeros(node.degree * (1,) + (node.dimension,), dtype=jnp.complex_)
        tensor = tensor.at[..., 0].set(1.0)
        return tensor

    return initializer


"""Returns a function that initializes kernels that all together
form a GHZ state. 'bloated' means that bond dimension of the
resulting state could be > 2 if necessary.
Args:
    key: jax random seed.
Returns:
    Initializer."""


def get_tensor_bloated_ghz_initializer(key: Array) -> Callable[[Node], Array]:
    def initializer(node: Node) -> Array:
        ghz_core = _gen_ghz_core(node.dimension, node.degree + 1)
        for bond_dim in reversed(node.bond_shape):
            if bond_dim < node.dimension:
                raise ValueError(
                    f"For GHZ initializer all bond indices must be >= the physical index, got one of the bond dimensions {bond_dim} and physical dimension {node.dimension}."
                )
            # Key is fixed for the purpose: for fixed shape one have to produce the same matrix
            bloater = normal(key, (bond_dim, node.dimension)).astype(jnp.complex64)
            bloater, _ = jnp.linalg.qr(bloater)
            ghz_core = jnp.tensordot(bloater, ghz_core, axes=[1, -1])
        assert ghz_core.shape == (
            *node.bond_shape,
            node.dimension,
        ), f"{ghz_core.shape}, {(*node.bond_shape, node.dimension)}"
        return ghz_core

    return initializer
