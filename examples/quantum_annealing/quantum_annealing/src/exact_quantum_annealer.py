import logging
from dataclasses import dataclass
from typing import Iterable, Tuple, Union, Dict, List, Callable, Optional
from jax import Array
from jax.random import split, uniform
import jax.numpy as jnp
import numpy as np
from qem import QuantumState
from .energy_function import EnergyFunction

log = logging.getLogger(__name__)

hadamard = np.sqrt(0.5) * np.array(
    [
        1,
        1,
        1,
        -1,
    ],
    dtype=np.complex128,
).reshape((2, 2))


@dataclass
class ExactQuantumAnnealingResults:
    configuration: Union[Array, None]
    density_matrices: Array
    density_matrices_history: Union[List[Array], None]


def _init_q2_gates(
    energy_function: EnergyFunction,
    tau: Union[float, Array],
) -> Dict[Tuple[int, int], Array]:
    gates = {}
    for pair, ampl in zip(
        energy_function.coupled_spin_pairs, energy_function.coupling_amplitudes
    ):
        degrees = energy_function.node_degrees
        degree1 = degrees[int(pair[0])]
        degree2 = degrees[int(pair[1])]
        ham = (
            ampl * jnp.array([1, -1, -1, 1], dtype=jnp.complex128)
            + (energy_function.fields[pair[0]] / degree1)
            * jnp.array([1, 1, -1, -1], dtype=jnp.complex128)
            + (energy_function.fields[pair[1]] / degree2)
            * jnp.array([1, -1, 1, -1], dtype=jnp.complex128)
        )
        gates[(int(pair[0]), int(pair[1]))] = np.array(
            jnp.diag(jnp.exp(-1j * tau * ham)).reshape((2, 2, 2, 2))
        )
    return gates


def _init_q1_gates(
    energy_function: EnergyFunction,
    tau: Union[float, Array],
) -> List[Array]:
    gate = jnp.array(
        [
            jnp.cos(tau),
            -1j * jnp.sin(tau),
            -1j * jnp.sin(tau),
            jnp.cos(tau),
        ]
    ).reshape((2, 2))
    return energy_function.fields.shape[0] * [np.array(gate)]


"""Executes a quantum annealer given an energy function, scheduler and a random seed."""


def run_exact_quantum_annealer(
    energy_function: EnergyFunction,
    scheduler: Iterable[Tuple[float, float]],
    key: Array,
    sample_measurements: bool,
    record_history: bool,
    state_callback: Optional[Callable[[QuantumState], None]] = None,
) -> ExactQuantumAnnealingResults:
    nodes_number = energy_function.nodes_number
    emulator = QuantumState(nodes_number)
    density_matrices_history = [] if record_history else None
    for node_id in range(nodes_number):
        emulator.apply1(node_id, hadamard)
    for layer_num, (mixing_time, coupling_time) in enumerate(scheduler):
        log.info(
            f"Running layer number {layer_num}, mixing_time: {mixing_time}, coupling_time: {coupling_time}"
        )
        coupling_gates = _init_q2_gates(energy_function, coupling_time)
        mixing_gates = _init_q1_gates(energy_function, mixing_time)
        for (node_id1, node_id2), coupling_gate in coupling_gates.items():
            emulator.apply2(node_id1, node_id2, coupling_gate)
        for node_id, mixing_gate in enumerate(mixing_gates):
            emulator.apply1(node_id, mixing_gate)
        if state_callback is not None:
            state_callback(emulator)
        if record_history:
            density_matrices = []
            for node_id in range(nodes_number):
                density_matrices.append(emulator.dens1(node_id))
            density_matrices_history.append(jnp.array(density_matrices))
    density_matrices = []
    for node_id in range(nodes_number):
        density_matrices.append(emulator.dens1(node_id))
    density_matrices = jnp.array(density_matrices)
    measurement_results = []
    if sample_measurements:
        for node_id in range(nodes_number):
            key, subkey = split(key)
            uniform_sample = np.array(uniform(subkey, (1,)))
            measurement_results.append(emulator.measure(node_id, uniform_sample))
        config = 1 - 2 * jnp.array(measurement_results)
    else:
        config = None
    return ExactQuantumAnnealingResults(
        config,
        density_matrices,
        density_matrices_history,
    )
