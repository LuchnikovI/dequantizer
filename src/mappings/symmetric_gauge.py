from typing import Iterator, Union, Dict, Callable
from ..graph import Node, Edge, NodeID, EdgeID
from jax import Array

"""Returns a function that turns Vidal gauge to the symmetric gauge.
Args:
    traverser: an iterator traversing elements of the tensor graph.
Returns:
    a function that takes node tensors and edge core diagonal tensors
    and returns updated node tensors with aggregated edge tensors."""


def get_symmetric_gauge_fixing_map(
    traverser: Iterator[Union[Node, Edge]]
) -> Callable[[Dict[NodeID, Array], Dict[EdgeID, Array]], Dict[NodeID, Array]]:
    raise NotImplementedError()
