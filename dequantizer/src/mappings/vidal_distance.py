from typing import Iterable, Union, Callable, Dict, List
import jax.numpy as jnp
from jax import Array
from ..graph import Node, Edge, EdgeID, NodeID
from .tensor_ops import vidal_dist

"""Returns a function that computes a distance to the vidal gauge.
The function takes the dict of node tensors and the dict of core edge diagonal tensors
and computes the distance to the Vidal gauge.
Args:
    traverser: an iterator that iterates over nodes and edges.
Returns:
    Vidal distance function."""


def get_vidal_gauge_distance_map(
    traverser: Iterable[Union[Node, Edge]]
) -> Callable[[Dict[NodeID, Array], Dict[EdgeID, Array]], Array]:
    def vidal_gauge_distance(
        tensors: Dict[NodeID, Array], core_edge_tensors: Dict[EdgeID, Array]
    ) -> Array:
        dist = jnp.array(0.0)
        edges_counter = 0
        for element in traverser:
            if isinstance(element, Node):
                neighboring_core_tensors: List[Array] = []
                for neighbor in element.neighbors:
                    neighbor_id = neighbor.id
                    if isinstance(neighbor_id, tuple):
                        neighboring_core_tensors.append(core_edge_tensors[neighbor_id])
                    else:
                        raise NotImplementedError(
                            "This branch is unreachable if the code is correct."
                        )
                dist += vidal_dist(tensors[element.id], neighboring_core_tensors)
            if isinstance(element, Edge):
                edges_counter += 1
        return dist / (2 * edges_counter)

    return vidal_gauge_distance
