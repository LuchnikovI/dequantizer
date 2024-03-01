import jax.numpy as jnp
from jax import Array


def _check_couplings_consistency(coupling_amplitudes: Array, coupled_spin_pairs: Array):
    if coupled_spin_pairs.shape[1] != 2:
        raise ValueError("Only two body interactions are allowed.")
    if coupling_amplitudes.shape[0] != coupled_spin_pairs.shape[0]:
        raise ValueError(
            "Mismatch between number of interaction amplitudes and number of interacting pairs."
        )


def _check_spins_number(coupled_spin_pairs: Array, fields: Array):
    spins_number_from_pairs = coupled_spin_pairs.max() + 1
    spins_number_from_fields = fields.shape[0]
    if spins_number_from_fields != spins_number_from_pairs:
        raise ValueError(
            "Number of magnetic fields is inconsistent with number of spins extracted from coupled pairs."
        )


def _check_coupled_spin_pairs(coupled_spin_pairs: Array):
    if not jnp.isdtype(coupled_spin_pairs.dtype, "signed integer"):
        raise ValueError("coupled_spins_pairs must be an array of integer values.")
    if coupled_spin_pairs.min() < 0:
        raise ValueError("Spin IDs must be non negative.")
