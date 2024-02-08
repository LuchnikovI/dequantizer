from typing import Union, Tuple
from jax import Array
from tensor_node_api import Node
from tensor_edge_api import Edge

"""Initializes a message given the message direction.
Args:
    message_direction: direction of the message.
Returns:
    message matrix."""
def message_random_nonnegative_initializer(message_dirrection: Union[Tuple[Node, Edge], Tuple[Edge, Node]]) -> Array:
    pass