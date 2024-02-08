from jax import Array
from tensor_node_api import Node

"""Initializes a tensor from i.i.d. complex normal distribution with 0 mean and std 1.
Args:
    node: node of a tensor graph.
Returns:
    tensor."""
def tensor_random_normal_initializer(node: Node) -> Array:
    pass

"""Initializes a tensor in such  way that the resulting tensor graph state is
|0>.
Args:
    node: node of a tensor graph.
Returns:
    tensor."""
def tensor_std_state_initializer(node: Node) -> Array:
    pass