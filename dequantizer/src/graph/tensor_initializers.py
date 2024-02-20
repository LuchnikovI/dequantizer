from typing import Callable
import jax.numpy as jnp
from jax.random import normal
from jax import Array
from .node import Node


def _gen_ghz_core(phys_dimension: int, modes_number: int) -> Array:
    if phys_dimension == 1:
        return jnp.array(1.0, dtype=jnp.complex128).reshape(modes_number * (1,))
    stride = (phys_dimension**modes_number - 1) // (phys_dimension - 1)
    elements_number = phys_dimension**modes_number
    core = jnp.zeros(elements_number, dtype=jnp.complex128)
    core = core.at[0:elements_number:stride].set(1.0)
    return core.reshape(modes_number * (phys_dimension,))


"""Returns an initializer that initializes a tensor from i.i.d. complex normal distribution with 0 mean and std 1.
Args:
    node: node of a tensor graph.
Returns:
    Initializer."""


def get_tensor_random_normal_initializer(key: Array) -> Callable[[Node], Array]:
    def initializer(node: Node) -> Array:
        shape = (*node.bond_shape, node.dimension, 2)
        tensor = normal(key, shape)
        tensor = tensor[..., 0] + 1j * tensor[..., 1]
        return tensor

    return initializer


"""Returns an initializer that initializes a tensor in such  way that the resulting tensor graph state is
|0>.
Args:
    node: node of a tensor graph.
Returns:
    Initializer."""


def get_tensor_std_state_initializer() -> Callable[[Node], Array]:
    def initializer(node: Node) -> Array:
        tensor = jnp.zeros(node.degree * (1,) + (node.dimension,), dtype=jnp.complex_)
        tensor = tensor.at[..., 0].set(1.0)
        return tensor

    return initializer


"""Returns a function that initializes kernels that all together
form a GHZ state. 'bloated' means that bond dimension of the
resulting state could be > 2 if necessary.
Args:
    key: jax random seed.
Returns:
    Initializer."""


def get_tensor_bloated_ghz_initializer(key: Array) -> Callable[[Node], Array]:
    def initializer(node: Node) -> Array:
        ghz_core = _gen_ghz_core(node.dimension, node.degree + 1)
        for bond_dim in reversed(node.bond_shape):
            if bond_dim < node.dimension:
                raise ValueError(
                    f"For GHZ initializer all bond indices must be >= the physical index, got one of the bond dimensions {bond_dim} and physical dimension {node.dimension}."
                )
            # Key is fixed for the purpose: for fixed shape one have to produce the same matrix
            bloater = normal(key, (bond_dim, node.dimension)).astype(jnp.complex128)
            bloater, _ = jnp.linalg.qr(bloater)
            ghz_core = jnp.tensordot(bloater, ghz_core, axes=[1, -1])
        assert ghz_core.shape == (
            *node.bond_shape,
            node.dimension,
        ), f"{ghz_core.shape}, {(*node.bond_shape, node.dimension)}"
        return ghz_core

    return initializer


"""Generates a node tensor that corresponds to a lattice whose
contraction with itself leads to the partition function
of a Potts model.
Args:
    two_spin_factor: a factor responsible for coupling between spins
        (must be symmetric);
    single_spin_factor: a factor responsible for energy of a single spin.
Return:
    Initializer."""


def get_potts_initializer(
    two_spin_factor: Array,
    single_spin_factor: Array,
) -> Callable[[Node], Array]:
    if len(single_spin_factor.shape) != 1:
        raise ValueError("A single spin factor must be a tensor of rank 1.")
    if len(two_spin_factor.shape) != 2:
        raise ValueError("A two spin factor must be a tensor of rank 2.")
    if two_spin_factor.shape[0] != two_spin_factor.shape[1]:
        raise ValueError("A two spin facto must have the same dimension per index.")
    if two_spin_factor.shape[0] != single_spin_factor.shape[0]:
        raise ValueError(
            "Index dimension of a single spin factor must match the dimension of two factor indices."
        )
    phys_dim = single_spin_factor.shape[0]
    single_spin_factor = jnp.sqrt(single_spin_factor)
    two_spin_factor = jnp.sqrt(two_spin_factor)
    lmbd, u = jnp.linalg.eigh(two_spin_factor)
    lmbd = jnp.array(lmbd, dtype=jnp.complex128)
    sqrt_two_spins_factor = (u * jnp.sqrt(lmbd)[jnp.newaxis]).T
    assert (
        jnp.linalg.norm(
            jnp.tensordot(sqrt_two_spins_factor, sqrt_two_spins_factor, axes=[0, 0])
            - two_spin_factor
        )
        < 1e-5
    )

    def initializer(node: Node) -> Array:
        if node.dimension != phys_dim:
            raise ValueError(
                "Node dimension must match with the factor index dimension."
            )
        tensor = single_spin_factor
        for neighbor in node.neighbors:
            if neighbor.dimension != phys_dim:
                raise ValueError(
                    "Bond dimensions of the node must match dimension of the factor indices."
                )
            tensor = tensor[..., jnp.newaxis, :] * sqrt_two_spins_factor
        assert tensor.shape == (node.degree + 1) * (
            phys_dim,
        ), f"{tensor.shape}, {(node.degree + 1) * (phys_dim,)}"
        return tensor

    return initializer
