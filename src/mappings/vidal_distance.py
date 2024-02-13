from typing import Iterator, Union, Callable, Dict
from ..graph import Node, Edge, EdgeID, NodeID
from jax import Array

"""Returns a function that computes a distance to the vidal gauge.
The function takes the dict of node tensors and the dict of core edge diagonal tensors
and computes the distance to the Vidal gauge.
Args:
    traverser: an iterator that iterates over nodes and edges.
Returns:
    Vidal distance function."""


def get_vidal_gauge_distance_map(
    traverser: Iterator[Union[Node, Edge]]
) -> Callable[[Dict[NodeID, Array], Dict[EdgeID, Array]], Array]:
    raise NotImplementedError()
