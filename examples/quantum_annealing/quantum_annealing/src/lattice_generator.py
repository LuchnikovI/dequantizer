from typing import Union, Optional
from jax import Array
from jax.random import split
from quantum_annealing.src.energy_function import (
    random_on_ibm_heavy_hex,
    random_on_one_heavy_hex_loop,
    random_on_small_tree,
    random_regular,
    random_regular_maxcut,
    EnergyFunction,
)


def generate_lattice(
    lattice_type: str,
    field_std: Union[None, float, Array],
    coupling_std: Union[None, float, Array],
    nodes_number: Optional[int],
    degree: Optional[int],
    key: Array,
) -> EnergyFunction:
    energy_function: EnergyFunction
    match lattice_type:

        case "ibm_heavy_hex":
            key, subkey = split(key)
            energy_function = random_on_ibm_heavy_hex(subkey, field_std, coupling_std)

        case "one_heavy_hex_loop":
            key, subkey = split(key)
            energy_function = random_on_one_heavy_hex_loop(
                subkey, field_std, coupling_std
            )

        case "small_tree":
            key, subkey = split(key)
            energy_function = random_on_small_tree(subkey, field_std, coupling_std)

        case "random_regular":
            key, subkey = split(key)
            assert degree is not None
            assert nodes_number is not None
            energy_function = random_regular(
                subkey, nodes_number, degree, field_std, coupling_std
            )

        case "random_regular_maxcut":
            key, subkey = split(key)
            assert degree is not None
            assert nodes_number is not None
            energy_function = random_regular_maxcut(
                subkey,
                nodes_number,
                degree,
            )

        case other:
            raise NotImplementedError(f"Lattice of type {other} is not implemented.")
    return energy_function
