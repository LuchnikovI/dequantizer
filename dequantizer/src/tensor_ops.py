from typing import List, Tuple, Optional, Union
import jax.numpy as jnp
from jax import Array
from jax.lax import round


def _find_rank(
    lmbd: Array, accuracy: Optional[Union[float, Array]], rank: Optional[int]
) -> Array:
    if accuracy is None:
        if rank is None:
            return jnp.array(lmbd.shape[0])
        return jnp.array(rank)
    scale = jnp.sqrt((lmbd**2).sum())
    return (
        lmbd.shape[0] - (jnp.sqrt(jnp.cumsum(lmbd[::-1] ** 2)) / scale < accuracy).sum()
    )


def _safe_inverse(lmbd: Array, eps: Union[Array, float]) -> Array:
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


def edge_svd(
    incoming_messages: List[Array], eps: Union[Array, float]
) -> Tuple[List[Array], Array]:
    if len(incoming_messages) != 2:
        raise NotImplementedError(
            "Edge svd is not yet implemented for edge degrees != 2."
        )
    uf, lf, _ = jnp.linalg.svd(incoming_messages[0], full_matrices=False)
    ub, lb, _ = jnp.linalg.svd(incoming_messages[1], full_matrices=False)
    m_sq_inv_f = uf @ (jnp.sqrt(_safe_inverse(lf, eps))[:, jnp.newaxis] * uf.conj().T)
    m_sq_inv_b = ub @ (jnp.sqrt(_safe_inverse(lb, eps))[:, jnp.newaxis] * ub.conj().T)
    m_sq_f = uf @ (jnp.sqrt(jnp.abs(lf))[:, jnp.newaxis] * uf.conj().T)
    m_sq_b = ub @ (jnp.sqrt(jnp.abs(lb))[:, jnp.newaxis] * ub.conj().T)
    u, lmbd, v = jnp.linalg.svd(
        jnp.tensordot(m_sq_f, m_sq_b, axes=[1, 1]), full_matrices=False
    )
    return [m_sq_inv_f @ u, m_sq_inv_b @ v.T], lmbd


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
        result_trace = jnp.trace(result)
        result /= result_trace
        rank = round(1 / jnp.abs(result[0, 0]))
        ideal = jnp.eye(result.shape[0])
        mask = (
            jnp.maximum(
                jnp.arange(ideal.shape[0])[:, jnp.newaxis],
                jnp.arange(ideal.shape[0])[jnp.newaxis],
            )
            < rank
        )
        ideal = ideal * mask
        ideal /= jnp.trace(ideal)
        _, lmbd, _ = jnp.linalg.svd(ideal - result, full_matrices=False)
        dist += jnp.sum(lmbd)
    return dist


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


def _decompose_gate(gate: Array, threshold: Union[Array, float]) -> Tuple[Array, Array]:
    assert len(gate.shape) == 4
    assert gate.shape[0] == gate.shape[2]
    assert gate.shape[1] == gate.shape[3]
    dim1 = gate.shape[0]
    dim2 = gate.shape[1]
    gate = jnp.transpose(gate, (0, 2, 1, 3)).reshape((dim1 * dim2, -1))
    u, lmbd, vh = jnp.linalg.svd(gate, full_matrices=False)
    rank = (lmbd > threshold).sum()
    lmbd_sqrt = jnp.sqrt(lmbd[:rank])
    controlling_half = jnp.transpose(
        (u[:, :rank] * lmbd_sqrt[jnp.newaxis]).reshape((dim1, dim1, rank)), (2, 0, 1)
    )
    controlled_half = (vh[:rank] * lmbd_sqrt[:, jnp.newaxis]).reshape(
        (rank, dim2, dim2)
    )
    return controlling_half, controlled_half


def _apply_half_gate(tensor: Array, half_gate: Array, index: int) -> Array:
    shape = list(tensor.shape)
    shape[index] *= half_gate.shape[0]
    tensor = jnp.tensordot(tensor, half_gate, axes=[-1, 2])
    shape_len = len(tensor.shape)
    tensor = jnp.transpose(
        tensor,
        [
            *range(index + 1),
            shape_len - 2,
            *range(index + 1, shape_len - 2),
            shape_len - 1,
        ],
    )
    tensor = tensor.reshape(shape)
    tensor /= jnp.linalg.norm(tensor)
    return tensor


"""Applies parts of decomposed gate to tensors.
Args:
    controlling_tensor: controlling tensor;
    controlled_tensor: controlled tensor;
    controlling_index: bond index number of the controlling tensor;
    controlled_index: bond index number of the controlled tensor;
    controlling_half: controlling half;
    controlled_half: controlled half.
Returns:
    updated first and second tensors."""


def apply_gate_halves(
    controlling_tensor: Array,
    controlled_tensor: Array,
    controlling_index: int,
    controlled_index: int,
    controlling_half: Array,
    controlled_half: Array,
) -> Tuple[Array, Array]:
    tensor1 = _apply_half_gate(controlling_tensor, controlling_half, controlling_index)
    tensor2 = _apply_half_gate(controlled_tensor, controlled_half, controlled_index)
    return tensor1, tensor2


def _simple_update_tensor_preprocessing(
    tensor: Array,
    lambdas: List[Array],
    half_gate: Array,
    threshold: Union[Array, float],
    index: int,
) -> Tuple[Array, Array]:
    degree = len(lambdas)
    assert len(tensor.shape) - 1 == degree
    for idx, lmbd in filter(lambda x: x[0] != index, enumerate(lambdas)):
        lmbd = lmbd.reshape(idx * (1,) + (lmbd.shape[0],) + (degree - idx) * (1,))
        tensor *= lmbd
    tensor = _apply_half_gate(tensor, half_gate, index)
    tensor = jnp.transpose(
        tensor,
        [
            *range(index),
            *range(index + 1, degree + 1),
            index,
        ],
    )
    shape = tensor.shape
    tensor = tensor.reshape((-1, shape[-1]))
    tensor, message = jnp.linalg.qr(tensor)
    tensor = tensor.reshape((*shape[:-1], -1))
    for idx, (_, lmbd) in enumerate(
        filter(lambda x: x[0] != index, enumerate(lambdas))
    ):
        lmbd = lmbd.reshape(idx * (1,) + (lmbd.shape[0],) + (degree - idx) * (1,))
        tensor *= _safe_inverse(lmbd, threshold)
    return tensor, message


"""Performs simple update of neighboring tensors under an action of two-sides gate.
Args:
    controlling_tensor: a tensor that is under affect of controlling index of a gate;
    controlled_tensor: a tensor that is under affect of controlled index of a gate;
    controlling_lambdas: singular values surrounding a controlling tensor; 
    controlled_lambdas: singular values surrounding a controlled tensor;
    controlling_half: controlling half of a gate;
    controlled_half: controlled half of a gate;
    controlling_index: controlling tensor index connected with the controlled tensor;
    controlled_index: controlled tensor index connected with the controlling tensor;
    threshold: a small value that is used for regularization purposes;
    accuracy: the maximal truncation error, not considered if None;
    max_rank: the maximal allowed edge dimension, not considered if None.
Returns:
    updated controlling tensor, updated controlled tensor, singular values vector connecting two tensors.
"""


def simple_update(
    controlling_tensor: Array,
    controlled_tensor: Array,
    controlling_lambdas: List[Array],
    controlled_lambdas: List[Array],
    controlling_half: Array,
    controlled_half: Array,
    controlling_index: int,
    controlled_index: int,
    threshold: Union[Array, float],
    accuracy: Optional[Union[float, Array]],
    max_rank: Optional[int] = None,
) -> Tuple[Array, Array, Array]:
    controlling_tensor, controlling_message = _simple_update_tensor_preprocessing(
        controlling_tensor,
        controlling_lambdas,
        controlling_half,
        threshold,
        controlling_index,
    )
    controlled_tensor, controlled_message = _simple_update_tensor_preprocessing(
        controlled_tensor,
        controlled_lambdas,
        controlled_half,
        threshold,
        controlled_index,
    )
    assert (
        jnp.abs(
            controlling_lambdas[controlling_index]
            - controlled_lambdas[controlled_index]
        )
        < 1e-5
    ).all()
    intermediate_lmbd = controlling_lambdas[controlling_index]
    additional_dim = controlling_message.shape[-1] // intermediate_lmbd.shape[0]
    intermediate_lmbd = jnp.tensordot(
        intermediate_lmbd, jnp.ones((additional_dim,)), axes=0
    ).reshape((-1,))
    core = jnp.tensordot(
        controlling_message * intermediate_lmbd, controlled_message, axes=[1, 1]
    )
    u, lmbd, vh = jnp.linalg.svd(core, full_matrices=False)
    rank = _find_rank(lmbd, None, max_rank)
    lmbd = lmbd[:rank]
    u = u[:, :rank]
    vh = vh[:rank]
    controlling_tensor = jnp.tensordot(controlling_tensor, u, axes=[-1, 0])
    controlling_degree = len(controlling_tensor.shape) - 1
    controlled_tensor = jnp.tensordot(controlled_tensor, vh, axes=[-1, 1])
    controlled_degree = len(controlled_tensor.shape) - 1
    controlling_tensor = jnp.transpose(
        controlling_tensor,
        [
            *range(controlling_index),
            controlling_degree,
            *range(controlling_index, controlling_degree),
        ],
    )
    controlled_tensor = jnp.transpose(
        controlled_tensor,
        [
            *range(controlled_index),
            controlled_degree,
            *range(controlled_index, controlled_degree),
        ],
    )
    return controlling_tensor, controlled_tensor, lmbd
