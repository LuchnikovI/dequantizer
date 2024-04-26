from dataclasses import dataclass
from typing import Union
import jax.numpy as jnp
import networkx as nx  # type: ignore
from jax import Array
from jax.random import split, normal

"""A class representing an energy function.
Fields:
    coupling_amplitudes: a one-dimensional array with amplitude coupling
        per pair of coupled spins;
    coupled_spin_pairs: a two-dimensional array whose zeroth index
        enumerates pairs and first index enumerates spins in a
        coupled pair;
    fields: magnetic field per spin."""


@dataclass
class EnergyFunction:
    coupling_amplitudes: Array
    coupled_spin_pairs: Array
    fields: Array


"""Generates an energy function corresponding to the classical
Ising model on the IBM Heavy Hex lattice consisting 127 spins with
external magnetic fields per spin and couplings sampled from
a normal distribution with zero mean.
Args:
    key: jax random seed;
    field_std: standard deviation of local fields distribution;
    coupling_std: standard deviation of couplings distribution.
Returns:
    EnergyFunction."""


def random_on_ibm_heavy_hex(
    key: Array,
    field_std: Union[Array, float] = 1.0,
    coupling_std: Union[Array, float] = 1.0,
) -> EnergyFunction:
    key, subkey = split(key)
    fields = normal(subkey, (127,)) * field_std
    coupled_spin_pairs = []
    for i in range(13):
        coupled_spin_pairs.append(jnp.array([i, i + 1]))
    for i in range(18, 32):
        coupled_spin_pairs.append(jnp.array([i, i + 1]))
    for i in range(37, 51):
        coupled_spin_pairs.append(jnp.array([i, i + 1]))
    for i in range(56, 70):
        coupled_spin_pairs.append(jnp.array([i, i + 1]))
    for i in range(75, 89):
        coupled_spin_pairs.append(jnp.array([i, i + 1]))
    for i in range(94, 108):
        coupled_spin_pairs.append(jnp.array([i, i + 1]))
    for i in range(113, 126):
        coupled_spin_pairs.append(jnp.array([i, i + 1]))
    coupled_spin_pairs.append(jnp.array([0, 14]))
    coupled_spin_pairs.append(jnp.array([14, 18]))
    coupled_spin_pairs.append(jnp.array([4, 15]))
    coupled_spin_pairs.append(jnp.array([15, 22]))
    coupled_spin_pairs.append(jnp.array([8, 16]))
    coupled_spin_pairs.append(jnp.array([16, 26]))
    coupled_spin_pairs.append(jnp.array([12, 17]))
    coupled_spin_pairs.append(jnp.array([17, 30]))
    coupled_spin_pairs.append(jnp.array([20, 33]))
    coupled_spin_pairs.append(jnp.array([33, 39]))
    coupled_spin_pairs.append(jnp.array([24, 34]))
    coupled_spin_pairs.append(jnp.array([34, 43]))
    coupled_spin_pairs.append(jnp.array([28, 35]))
    coupled_spin_pairs.append(jnp.array([35, 47]))
    coupled_spin_pairs.append(jnp.array([32, 36]))
    coupled_spin_pairs.append(jnp.array([36, 51]))
    coupled_spin_pairs.append(jnp.array([37, 52]))
    coupled_spin_pairs.append(jnp.array([52, 56]))
    coupled_spin_pairs.append(jnp.array([41, 53]))
    coupled_spin_pairs.append(jnp.array([53, 60]))
    coupled_spin_pairs.append(jnp.array([45, 54]))
    coupled_spin_pairs.append(jnp.array([54, 64]))
    coupled_spin_pairs.append(jnp.array([49, 55]))
    coupled_spin_pairs.append(jnp.array([55, 68]))
    coupled_spin_pairs.append(jnp.array([58, 71]))
    coupled_spin_pairs.append(jnp.array([71, 77]))
    coupled_spin_pairs.append(jnp.array([62, 72]))
    coupled_spin_pairs.append(jnp.array([72, 81]))
    coupled_spin_pairs.append(jnp.array([66, 73]))
    coupled_spin_pairs.append(jnp.array([73, 85]))
    coupled_spin_pairs.append(jnp.array([70, 74]))
    coupled_spin_pairs.append(jnp.array([74, 89]))
    coupled_spin_pairs.append(jnp.array([75, 90]))
    coupled_spin_pairs.append(jnp.array([90, 94]))
    coupled_spin_pairs.append(jnp.array([79, 91]))
    coupled_spin_pairs.append(jnp.array([91, 98]))
    coupled_spin_pairs.append(jnp.array([83, 92]))
    coupled_spin_pairs.append(jnp.array([92, 102]))
    coupled_spin_pairs.append(jnp.array([87, 93]))
    coupled_spin_pairs.append(jnp.array([93, 106]))
    coupled_spin_pairs.append(jnp.array([96, 109]))
    coupled_spin_pairs.append(jnp.array([109, 114]))
    coupled_spin_pairs.append(jnp.array([100, 110]))
    coupled_spin_pairs.append(jnp.array([110, 118]))
    coupled_spin_pairs.append(jnp.array([104, 111]))
    coupled_spin_pairs.append(jnp.array([111, 122]))
    coupled_spin_pairs.append(jnp.array([108, 112]))
    coupled_spin_pairs.append(jnp.array([112, 126]))
    key, subkey = split(key)
    coupling_amplitudes = normal(subkey, (len(coupled_spin_pairs),)) * coupling_std
    return EnergyFunction(
        coupling_amplitudes,
        jnp.array(coupled_spin_pairs),
        fields,
    )


"""Generates an energy function corresponding to the classical
Ising model on a single Heavy Hex loop with
external magnetic fields per spin and couplings sampled from
a normal distribution with zero mean.
Args:
    key: jax random seed;
    field_std: standard deviation of local fields distribution;
    coupling_std: standard deviation of couplings distribution.
Returns:
    EnergyFunction."""


def random_on_one_heavy_hex_loop(
    key: Array,
    field_std: Union[Array, float] = 1.0,
    coupling_std: Union[Array, float] = 1.0,
) -> EnergyFunction:
    key, subkey = split(key)
    fields = normal(subkey, (12,)) * field_std
    coupled_spin_pairs = []
    for i in range(4):
        coupled_spin_pairs.append(jnp.array([i, i + 1]))
    for i in range(5, 9):
        coupled_spin_pairs.append(jnp.array([i, i + 1]))
    coupled_spin_pairs.append(jnp.array([0, 10]))
    coupled_spin_pairs.append(jnp.array([10, 5]))
    coupled_spin_pairs.append(jnp.array([4, 11]))
    coupled_spin_pairs.append(jnp.array([11, 9]))
    coupling_amplitudes = normal(subkey, (len(coupled_spin_pairs),)) * coupling_std
    return EnergyFunction(
        coupling_amplitudes,
        jnp.array(coupled_spin_pairs),
        fields,
    )


def random_on_small_tree(
    key: Array,
    field_std: Union[Array, float] = 1.0,
    coupling_std: Union[Array, float] = 1.0,
) -> EnergyFunction:
    key, subkey = split(key)
    fields = normal(subkey, (10,)) * field_std
    coupled_spin_pairs = []
    coupled_spin_pairs.append(jnp.array([0, 1]))
    coupled_spin_pairs.append(jnp.array([0, 2]))
    coupled_spin_pairs.append(jnp.array([0, 3]))
    coupled_spin_pairs.append(jnp.array([1, 4]))
    coupled_spin_pairs.append(jnp.array([1, 5]))
    coupled_spin_pairs.append(jnp.array([2, 6]))
    coupled_spin_pairs.append(jnp.array([2, 7]))
    coupled_spin_pairs.append(jnp.array([3, 8]))
    coupled_spin_pairs.append(jnp.array([3, 9]))
    coupling_amplitudes = normal(subkey, (len(coupled_spin_pairs),)) * coupling_std
    return EnergyFunction(
        coupling_amplitudes,
        jnp.array(coupled_spin_pairs),
        fields,
    )


def random_regular(
    key: Array,
    nodes_number: int = 150,
    degree: int = 3,
    field_std: Union[Array, float] = 1.0,
    coupling_std: Union[Array, float] = 1.0,
) -> EnergyFunction:
    key, subkey = split(key)
    graph = nx.random_regular_graph(degree, nodes_number, int(subkey[0]))
    key, subkey = split(key)
    fields = normal(subkey, (nodes_number,)) * field_std
    coupled_spin_pairs = []
    for edge in graph.edges:
        coupled_spin_pairs.append(jnp.array(edge))
    key, subkey = split(key)
    coupling_amplitudes = normal(subkey, (len(coupled_spin_pairs),)) * coupling_std
    return EnergyFunction(
        coupling_amplitudes,
        jnp.array(coupled_spin_pairs),
        fields,
    )


def random_regular_maxcut(
    key: Array,
    nodes_number: int = 150,
    degree: int = 3,
) -> EnergyFunction:
    key, subkey = split(key)
    graph = nx.random_regular_graph(degree, nodes_number, int(subkey[0]))
    fields = jnp.zeros((nodes_number,))
    coupled_spin_pairs = []
    for edge in graph.edges:
        coupled_spin_pairs.append(jnp.array(edge))
    coupling_amplitudes = -jnp.ones((len(coupled_spin_pairs),))
    return EnergyFunction(
        coupling_amplitudes,
        jnp.array(coupled_spin_pairs),
        fields,
    )
