from typing import Union, Iterator, Callable, Dict, Tuple, List
from jax import Array
from ..graph import (
    Node,
    NodeID,
    Edge,
    MessageID,
    EdgeID,
)
from .tensor_ops import edge_svd, absorb_by_node

"""Returns a function that fixes a tensor graph gauge to the Vidal gauge.
Args:
    traverser: iterator over elements of the tensor graph.
Returns:
    a function that takes node tensors dict and messages dict
    and returns an updated tensors dict and edge diagonal tensors."""


def get_vidal_gauge_fixing_map(traverser: Iterator[Union[Node, Edge]]) -> Callable[
    [Dict[NodeID, Array], Dict[MessageID, Array]],
    Tuple[Dict[NodeID, Array], Dict[EdgeID, Array]],
]:
    raise NotImplementedError()
