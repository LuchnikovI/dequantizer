from typing import List, Dict, Union, Iterator, Callable
from jax import Array
from ..graph import (
    Node,
    NodeID,
    Edge,
    MessageID,
)
from .tensor_ops import pass_through_node, pass_through_edge

"""Returns a function that performs one iteration of belief propagation.
Args:
    traverser: an iterator that traverse elements of a tensor graph (nodes and edges).
Returns:
    A function that performs one iteration of belief propagation. It takes
        the list of tensors, dict with messages and returns updated messages.
Notes:
    Messages are always normalized by 1, i.e. |m|_F = 1. Therefore, they cannot be used
    for estimation of the graph tensor network normalization. 
"""


def get_belief_propagation_map(
    traverser: Iterator[Union[Node, Edge]]
) -> Callable[[Dict[NodeID, Array], Dict[MessageID, Array]], Dict[MessageID, Array]]:
    raise NotImplementedError()
