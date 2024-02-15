from typing import List, Tuple
import jax.numpy as jnp
from jax import Array


def _safe_inverse(lmbd: Array, eps: Array) -> Array:
    return ((lmbd < eps) * jnp.finfo(jnp.float32).max + lmbd) ** -1


"""Performs update of messages coming to a node.
Args:
    tensor: a node tensor;
    incoming_messages: messages coming to the node.
Returns:
    updated messages."""


def pass_through_node(tensor: Array, incoming_messages: List[Array]) -> List[Array]:
    degree = len(tensor.shape) - 1
    updated_messages: List[Array] = []
    assert degree == len(incoming_messages), f"{degree}, {len(incoming_messages)}"
    conj_tensor = tensor.conj()
    for idx in range(degree):
        up_part = tensor
        for i, message in enumerate(incoming_messages):
            if i != idx:
                up_part = jnp.tensordot(up_part, message, axes=[0, 1])
            else:
                up_part = jnp.transpose(up_part, [*range(1, degree + 1), 0])
        up_part = jnp.transpose(
            up_part, [idx + 1, *range(idx + 1), *range(idx + 2, degree + 1)]
        )
        down_part = jnp.transpose(
            conj_tensor, [idx, degree, *range(idx), *range(idx + 1, degree)]
        )
        assert up_part.shape == down_part.shape, f"{up_part.shape}, {down_part.shape}"
        up_part = up_part.reshape((up_part.shape[0], -1))
        down_part = down_part.reshape((down_part.shape[0], -1))
        updated_message = jnp.tensordot(down_part, up_part, axes=[1, 1])
        updated_message /= jnp.linalg.norm(updated_message)
        updated_messages.append(updated_message)
    return updated_messages


"""Performs update of messages coming to an edge.
Args:
    incoming_messages: messages coming to the node.
Returns:
    updated messages."""


def pass_through_edge(incoming_messages: List[Array]) -> List[Array]:
    if len(incoming_messages) != 2:
        NotImplementedError(
            "Passing messages trough egdes with degree != 2 is not yet implemented."
        )
    updated_messages = [incoming_messages[1], incoming_messages[0]]
    return updated_messages


"""Absorbs matrices fixing the gauge by the node tensor.
Args:
    tensor: a node tensor;
    matrices: matrices fixing the gauge.
Returns:
    updated tensor.
"""


def absorb_by_node(tensor: Array, matrices: List[Array]) -> Array:
    for matrix in matrices:
        tensor = jnp.tensordot(tensor, matrix, axes=[0, 0])
    tensor = jnp.transpose(tensor, [*range(1, len(tensor.shape)), 0])
    return tensor


"""Performs globally valid svd of en edge given the set of incoming messages
to the edge.
Args:
    incoming_messages: set of edge incoming messages;
    eps: a threshold applied to truncate a singular value if it is too small.
Returns:
    set of matrices orthogonalizing neighboring branches and the diagonal core tensor of the edge.
Note:
    Currently works only for edges of degree 2. It is not clear yet how to generalized it
    on the case with higher degrees (Tucker / HOSVD / etc?)."""


def edge_svd(incoming_messages: List[Array], eps: Array) -> Tuple[List[Array], Array]:
    if len(incoming_messages) != 2:
        raise NotImplementedError(
            "Edge svd is not yet implemented for edge degrees != 2."
        )
    uf, lf, _ = jnp.linalg.svd(incoming_messages[0])
    ub, lb, _ = jnp.linalg.svd(incoming_messages[1])
    m_sq_inv_f = uf @ (jnp.sqrt(_safe_inverse(lf, eps))[:, jnp.newaxis] * uf.conj().T)
    m_sq_inv_b = ub @ (jnp.sqrt(_safe_inverse(lb, eps))[:, jnp.newaxis] * ub.conj().T)
    m_sq_f = uf @ (jnp.sqrt(jnp.abs(lf))[:, jnp.newaxis] * uf.conj().T)
    m_sq_b = ub @ (jnp.sqrt(jnp.abs(lb))[:, jnp.newaxis] * ub.conj().T)
    u, lmbd, _ = jnp.linalg.svd(m_sq_f @ m_sq_b.T)
    return [m_sq_inv_f @ u, m_sq_inv_b @ u.conj()], lmbd


"""Computes Vidal distance for a single tensor.
Args:
    tensor: a tensor;
    neighboring_core_edge_tensors: neighboring core edge tensors.
Returns:
    A distance."""


def vidal_dist(tensor: Array, neighboring_core_edge_tensors: List[Array]) -> Array:
    degree = len(tensor.shape) - 1
    assert degree == len(neighboring_core_edge_tensors)
    dist = jnp.array(0.0)
    conj_tensor = tensor.conj()
    for idx in range(degree):
        up_part = tensor
        for i, lmbd in filter(
            lambda x: x[0] != idx, enumerate(neighboring_core_edge_tensors)
        ):
            lmbd_sq = (
                lmbd.reshape(i * (1,) + (lmbd.shape[0],) + (degree - i) * (1,)) ** 2
            )
            up_part *= lmbd_sq
        up_part = jnp.transpose(
            up_part, [idx, *range(idx), *range(idx + 1, degree + 1)]
        )
        up_part = up_part.reshape((up_part.shape[0], -1))
        down_part = jnp.transpose(
            conj_tensor, [idx, *range(idx), *range(idx + 1, degree + 1)]
        )
        down_part = down_part.reshape((down_part.shape[0], -1))
        result = jnp.tensordot(down_part, up_part, axes=[1, 1])
        result /= result[0, 0]
        rank = (jnp.diag(result) > 0.5).sum()
        ideal = jnp.zeros(result.shape)
        ideal = ideal.at[:rank, :rank].set(jnp.eye(rank))
        dist += jnp.linalg.norm(result - ideal)
    return dist / degree


"""Multiply a given tensor by sqrt of lambdas from all sides. It is necessary to
bring the tensor graph to the symmetric canonical form.
Args:
    tensor: a tensor,
    lambdas: lambda vectors;
Returns:
    canonicalized tensor."""


def canonicalize(tensor: Array, lambdas: List[Array]) -> Array:
    degree = len(lambdas)
    assert degree == len(tensor.shape) - 1
    for idx, lmbd in enumerate(lambdas):
        lmbd = lmbd.reshape(idx * (1,) + (lmbd.shape[0],) + (degree - idx) * (1,))
        tensor *= jnp.sqrt(lmbd)
    return tensor
