import jax.numpy as jnp
from jax import Array
from .energy_function import EnergyFunction

"""Evaluates an energy given the energy function and a config."""


def eval_energy(energy_function: EnergyFunction, config: Array) -> float:
    external_field_energy = jnp.tensordot(energy_function.fields, config, axes=1)
    coupling_energy = jnp.tensordot(
        config[energy_function.coupled_spin_pairs[:, 0]]
        * config[energy_function.coupled_spin_pairs[:, 1]],
        energy_function.coupling_amplitudes,
        axes=1,
    )
    return -external_field_energy - coupling_energy
