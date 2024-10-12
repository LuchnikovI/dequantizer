from typing import Optional
import logging
from dataclasses import dataclass
from typing import Iterable, Tuple, Union, Dict, List
from jax import Array
import jax.numpy as jnp
from dequantizer import TensorGraph, BPQuantumEmulator
from .energy_function import EnergyFunction

log = logging.getLogger(__name__)

hadamard = jnp.sqrt(0.5) * jnp.array(
    [
        1,
        1,
        1,
        -1,
    ],
    dtype=jnp.complex128,
).reshape((2, 2))


@dataclass
class QuantumAnnealingResults:
    configuration: Union[Array, None]
    density_matrices: Array
    density_matrices_history: Union[List[Array], None]
    vidal_distances_after_regauging: Array
    truncation_affected_vidal_distances: Array
    truncation_errors: Array
    entropies: Array


def _tensor_graph_init(energy_function: EnergyFunction) -> TensorGraph:
    tensor_graph = TensorGraph()
    qubits_number = energy_function.fields.shape[0]
    for _ in range(qubits_number):
        tensor_graph.add_node()
    for pair in energy_function.coupled_spin_pairs:
        tensor_graph.add_edge((int(pair[0]), int(pair[1])), 1)
    return tensor_graph


def _init_q2_gates(
    energy_function: EnergyFunction,
    tensor_graph: TensorGraph,
    tau: Union[float, Array],
) -> Dict[Tuple[int, int], Array]:
    gates = {}
    for pair, ampl in zip(
        energy_function.coupled_spin_pairs, energy_function.coupling_amplitudes
    ):
        node1 = tensor_graph.get_node(int(pair[0]))
        node2 = tensor_graph.get_node(int(pair[1]))
        if node1 is None or node2 is None:
            raise ValueError("A node is not found, more likely it is a bug")
        degree1 = node1.degree
        degree2 = node2.degree
        ham = (
            ampl * jnp.array([1, -1, -1, 1], dtype=jnp.complex128)
            + (energy_function.fields[pair[0]] / degree1)
            * jnp.array([1, 1, -1, -1], dtype=jnp.complex128)
            + (energy_function.fields[pair[1]] / degree2)
            * jnp.array([1, -1, 1, -1], dtype=jnp.complex128)
        )
        gates[(int(pair[0]), int(pair[1]))] = jnp.diag(
            jnp.exp(-1j * tau * ham)
        ).reshape((2, 2, 2, 2))
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
    return energy_function.fields.shape[0] * [gate]


"""Executes a quantum annealer given an energy function, scheduler and a random seed."""


def run_quantum_annealer(
    energy_function: EnergyFunction,
    scheduler: Iterable[Tuple[float, float]],
    key: Array,
    max_chi: int,
    gates_number_per_regauging: int,
    accuracy: float,
    max_belief_propagation_iteration: int,
    synchronous_update: bool,
    traversal_type: str,
    sample_measurements: bool,
    record_history: bool,
) -> QuantumAnnealingResults:
    tensor_graph = _tensor_graph_init(energy_function)
    emulator = BPQuantumEmulator(
        tensor_graph,
        key,
        max_chi,
        gates_number_per_regauging,
        accuracy,
        max_belief_propagation_iteration,
        synchronous_update,
        traversal_type,
    )
    density_matrices_history: Optional[List[Array]] = [] if record_history else None
    for node_id in range(tensor_graph.nodes_number):
        emulator.apply_q1(hadamard, node_id)
    for layer_num, (mixing_time, coupling_time) in enumerate(scheduler):
        log.info(
            f"Running layer number {layer_num}, mixing_time: {mixing_time}, coupling_time: {coupling_time}"
        )
        coupling_gates = _init_q2_gates(energy_function, tensor_graph, coupling_time)
        mixing_gates = _init_q1_gates(energy_function, mixing_time)
        for (node_id1, node_id2), coupling_gate in coupling_gates.items():
            emulator.apply_q2(coupling_gate, node_id1, node_id2)
        for node_id, mixing_gate in enumerate(mixing_gates):
            emulator.apply_q1(mixing_gate, node_id)
        if record_history:
            density_matrices = []
            for node_id in range(tensor_graph.nodes_number):
                density_matrices.append(emulator.dens_q1(node_id))
            assert not (density_matrices_history is None)
            density_matrices_history.append(jnp.array(density_matrices))
    density_matrices = []
    for node_id in range(tensor_graph.nodes_number):
        density_matrices.append(emulator.dens_q1(node_id))
    density_matrices_arr = jnp.array(density_matrices)
    measurement_results = []
    if sample_measurements:
        for node_id in range(tensor_graph.nodes_number):
            measurement_results.append(emulator.measure(node_id))
        config = 1 - 2 * jnp.array(measurement_results)
    else:
        config = None
    return QuantumAnnealingResults(
        config,
        density_matrices_arr,
        density_matrices_history,
        emulator.after_regauging_vidal_distances,
        emulator.truncated_affected_vidal_distances,
        jnp.array(emulator.truncation_errors),
        jnp.array(emulator.entropies),
    )
