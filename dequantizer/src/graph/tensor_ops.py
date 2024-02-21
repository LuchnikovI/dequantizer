from typing import Tuple, Union
import jax.numpy as jnp
from jax import Array


def _decompose_gate(gate: Array, threshold: Union[Array, float]) -> Tuple[Array, Array]:
    assert len(gate.shape) == 4
    assert gate.shape[0] == gate.shape[2]
    assert gate.shape[1] == gate.shape[3]
    dim1 = gate.shape[0]
    dim2 = gate.shape[1]
    gate = jnp.transpose(gate, (0, 2, 1, 3)).reshape((dim1 * dim2, -1))
    u, lmbd, vh = jnp.linalg.svd(gate)
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
