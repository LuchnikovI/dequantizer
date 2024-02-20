from typing import Dict, List
import jax.numpy as jnp
from jax import Array
from ..graph.node import Node, NodeID
from ..graph.element import MessageID

"""Returns a Belief Propagation based estimation of a local density matrix.
Args:
    node: a node where to estimate a density matrix;
    tensors: node tensors;
    messages: messages after convergence of the Belief Propagation iterations.
Returns:
    a local density matrix."""


def get_one_side_density_matrix(
    node: Node,
    tensors: Dict[NodeID, Array],
    messages: Dict[MessageID, Array],
) -> Array:
    degree = node.degree
    tensor = tensors[node.id]
    assert degree == len(tensor.shape) - 1
    conj_tensor = tensor.conj()
    conj_tensor = jnp.transpose(conj_tensor, [degree, *range(degree)]).reshape(
        (tensor.shape[-1], -1)
    )
    for neighbor in node.neighbors:
        message = messages[MessageID(neighbor.id, node.id)]
        tensor = jnp.tensordot(tensor, message, axes=[0, 1])
    tensor = tensor.reshape((tensor.shape[-1], -1))
    dens = jnp.tensordot(tensor, conj_tensor, axes=[1, 1])
    dens /= jnp.trace(dens)
    return dens
