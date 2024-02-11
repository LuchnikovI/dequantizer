from typing import Callable
import jax.numpy as jnp
from jax.random import normal
from jax import Array
from node import Node

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
        tensor = tensor.at[..., 0].set(1.)
        return tensor
    return initializer
