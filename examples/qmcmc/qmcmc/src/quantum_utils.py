from typing import Dict, Tuple, Union, List
import jax.numpy as jnp
from jax import Array
from dequantizer import TensorGraph, Node, NodeID
from .base_annealer import EnergyFunction


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
    alpha: Union[float, Array],
    gamma: Union[float, Array],
) -> Dict[Tuple[int, int], Array]:
    step = alpha * (1 - gamma) * tau
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
            step * ampl * jnp.array([1, -1, -1, 1], dtype=jnp.complex128)
            + (step * energy_function.fields[pair[0]] / degree1)
            * jnp.array([1, 1, -1, -1], dtype=jnp.complex128)
            + (step * energy_function.fields[pair[1]] / degree2)
            * jnp.array([1, -1, 1, -1], dtype=jnp.complex128)
        )
        gates[(int(pair[0]), int(pair[1]))] = jnp.diag(jnp.exp(1j * ham)).reshape(
            (2, 2, 2, 2)
        )
    return gates


def _init_q1_gates(
    energy_function: EnergyFunction,
    tau: Union[float, Array],
    gamma: Union[float, Array],
) -> List[Array]:
    step = tau * gamma
    gate = jnp.array(
        [
            jnp.cos(step),
            -1j * jnp.sin(step),
            -1j * jnp.sin(step),
            jnp.cos(step),
        ]
    ).reshape((2, 2))
    return energy_function.fields.shape[0] * [gate]


def _compute_alpha(energy_function: EnergyFunction) -> float:
    num = jnp.sqrt(energy_function.fields.shape[0])
    den = jnp.sqrt(
        (energy_function.fields**2).sum()
        + (energy_function.coupling_amplitudes**2).sum()
    )
    return float(num / den)
