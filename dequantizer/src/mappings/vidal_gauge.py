from typing import Union, Iterable, Callable, Dict, Tuple, List
from jax import Array
from ..graph import (
    Node,
    NodeID,
    Edge,
    MessageID,
    EdgeID,
)
from ..tensor_ops import edge_svd, absorb_by_node

"""Returns a function that fixes a tensor graph gauge to the Vidal gauge.
Args:
    traverser: iterator over elements of the tensor graph;
    eps: a threshold of zeroing a singular value if it is too small.
Returns:
    a function that takes node tensors dict and messages dict
    and returns an updated tensors dict and edge diagonal tensors."""


def get_vidal_gauge_fixing_map(
    traverser: Iterable[Union[Node, Edge]], eps: Array
) -> Callable[
    [Dict[NodeID, Array], Dict[MessageID, Array]],
    Tuple[Dict[NodeID, Array], Dict[EdgeID, Array]],
]:
    def vidal_gauge_fixing_map(
        tensors: Dict[NodeID, Array], messages: Dict[MessageID, Array]
    ) -> Tuple[Dict[NodeID, Array], Dict[EdgeID, Array]]:
        core_edge_tensors: Dict[EdgeID, Array] = {}
        gauged_tensors: Dict[NodeID, Array] = {}
        orthogonalizers: Dict[MessageID, Array] = {}
        nodes: List[Node] = []
        for element in traverser:
            incoming_messages: List[Array] = []
            if isinstance(element, Edge):
                for neighbor in element.neighbors:
                    message_id = MessageID(src=neighbor.id, dst=element.id)
                    incoming_messages.append(messages[message_id])
                local_orthogonalizers, core_edge_tensor = edge_svd(
                    incoming_messages, eps
                )
                if isinstance(element.id, tuple):
                    core_edge_tensors[element.id] = core_edge_tensor
                for orthogonalizer, neighbor in zip(
                    local_orthogonalizers, element.neighbors
                ):
                    orthogonalizers[MessageID(src=element.id, dst=neighbor.id)] = (
                        orthogonalizer
                    )
            else:
                nodes.append(element)
        for node in nodes:
            local_orthogonalizers = []
            for neighbor in node.neighbors:
                orthogonalizer_id = MessageID(src=neighbor.id, dst=node.id)
                local_orthogonalizers.append(orthogonalizers[orthogonalizer_id])
            gauged_tensor = absorb_by_node(tensors[node.id], local_orthogonalizers)
            gauged_tensors[node.id] = gauged_tensor
        return gauged_tensors, core_edge_tensors

    return vidal_gauge_fixing_map
