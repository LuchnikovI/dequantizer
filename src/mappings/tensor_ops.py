from typing import List, Tuple
from jax import Array

"""Performs update of messages coming to a node.
Args:
    tensor: a node tensor;
    incoming_messages: messages coming to the node.
Returns:
    updated messages."""


def pass_through_node(tensor: Array, incoming_messages: List[Array]) -> List[Array]:
    raise NotImplementedError()


"""Performs update of messages coming to an edge.
Args:
    incoming_messages: messages coming to the node.
Returns:
    updated messages."""


def pass_through_edge(incoming_messages: List[Array]) -> List[Array]:
    raise NotImplementedError()


"""Absorbs matrices fixing the gauge by the node tensor.
Args:
    tensor: a node tensor;
    matrices: matrices fixing the gauge.
Returns:
    updated tensor.
"""


def absorb_by_node(tensor: Array, matrices: List[Array]) -> Array:
    raise NotImplementedError()


"""Performs globally valid svd of en edge given the set of incoming messages
to the edge.
Args:
    incoming_messages: set of edge incoming messages.
Returns:
    set of matrices orthogonalizing neighboring branches and the diagonal core tensor of the edge.
Note:
    Currently works only for edges of degree 2. It is not clear yet how to generalized it
    on the case with higher degrees (Tucker / HOSVD / etc?)."""


def edge_svd(incoming_messages: List[Array]) -> Tuple[List[Array], Array]:
    raise NotImplementedError()
