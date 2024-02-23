from typing import List, Dict, Union, Iterable, Callable
from jax import Array
from ..graph import (
    Node,
    NodeID,
    Edge,
    MessageID,
)
from ..tensor_ops import pass_through_node, pass_through_edge

"""Returns a function that performs one iteration of belief propagation.
Args:
    traverser: an iterator that traverse elements of a tensor graph (nodes and edges).
    is_synchronous: a flag specifying if the update is synchronous;
Returns:
    A function that performs one iteration of belief propagation. It takes
        the list of tensors, dict with messages and returns updated messages.
Notes:
    Messages are always normalized by 1, i.e. |m|_F = 1. Therefore, they cannot be used
    for estimation of the graph tensor network normalization. 
"""


def get_belief_propagation_map(
    traverser: Iterable[Union[Node, Edge]],
    is_synchronous: bool = False,
) -> Callable[[Dict[NodeID, Array], Dict[MessageID, Array]], Dict[MessageID, Array]]:
    def gauge_fixing_map(
        tensors: Dict[NodeID, Array], messages: Dict[MessageID, Array]
    ) -> Dict[MessageID, Array]:
        updated_messages: Dict[MessageID, Array] = {}
        for element in traverser:
            neighboring_messages: List[Array] = []
            for neighbor in element.neighbors:
                message_id = MessageID(src=neighbor.id, dst=element.id)
                neighboring_message: Array
                if is_synchronous:
                    neighboring_message = messages[message_id]
                else:
                    trial_updated_message = updated_messages.get(message_id)
                    if trial_updated_message is None:
                        neighboring_message = messages[message_id]
                    else:
                        neighboring_message = trial_updated_message
                neighboring_messages.append(neighboring_message)
            updated_neighboring_messages: List[Array]
            if isinstance(element, Node):
                updated_neighboring_messages = pass_through_node(
                    tensors[element.id], neighboring_messages
                )
            elif isinstance(element, Edge):
                updated_neighboring_messages = pass_through_edge(neighboring_messages)
            else:
                raise NotImplementedError(
                    "This branch is unreachable if the code is correct."
                )
            for neighbor, message in zip(
                element.neighbors, updated_neighboring_messages
            ):
                message_id = MessageID(src=element.id, dst=neighbor.id)
                updated_messages[message_id] = message
        return updated_messages

    return gauge_fixing_map
