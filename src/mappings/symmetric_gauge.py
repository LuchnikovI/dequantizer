from typing import Iterable, Union, Dict, Callable, List
from jax import Array
from ..graph import Node, Edge, NodeID, EdgeID
from .tensor_ops import canonicalize

"""Returns a function that turns Vidal gauge to the symmetric gauge.
Args:
    traverser: an iterator traversing elements of the tensor graph.
Returns:
    a function that takes node tensors and edge core diagonal tensors
    and returns updated node tensors with aggregated edge tensors."""


def get_symmetric_gauge_fixing_map(
    traverser: Iterable[Union[Node, Edge]]
) -> Callable[[Dict[NodeID, Array], Dict[EdgeID, Array]], Dict[NodeID, Array]]:
    def symmetric_gauge(
        tensors: Dict[NodeID, Array], core_edge_tensors: Dict[EdgeID, Array]
    ) -> Dict[NodeID, Array]:
        canonicalized_tensors: Dict[NodeID, Array] = {}
        for element in traverser:
            if isinstance(element, Node):
                tensor = tensors[element.id]
                lambdas: List[Array] = []
                for neighbor in element.neighbors:
                    neighbor_id = neighbor.id
                    if isinstance(neighbor_id, tuple):
                        lambdas.append(core_edge_tensors[neighbor_id])
                    else:
                        raise NotImplementedError(
                            "This branch is unreachable if the code is correct."
                        )
                canonicalized_tensors[element.id] = canonicalize(tensor, lambdas)
        return canonicalized_tensors

    return symmetric_gauge
