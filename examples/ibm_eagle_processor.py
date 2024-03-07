#!/usr/bin/env python3

"""In this example we simulate dynamics of 62-nd qubit (a qubit in the middle
of a lattice) of the Eagle IBM quantum processor either for 127 qubits
(the original IBM processor) or for its infinite modification.

Note, that simulation for an infinite processor is way faster due to 
the translational symmetry."""

import os

os.environ["JAX_ENABLE_X64"] = "True"

import jax

jax.config.update("jax_platform_name", "cpu")

import matplotlib

matplotlib.rcParams["font.family"] = "monospace"

import logging
from typing import Union, List, Optional
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey
from dequantizer import (
    BPQuantumEmulator,
    get_heavy_hex_ibm_eagle_lattice,
    get_heavy_hex_ibm_eagle_lattice_infinite,
    Edge,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

"""This function runs dynamics simulation for either 127 qubits or
infinite IBM Eagle processor (see https://arxiv.org/abs/2306.14887).
Args:
    max_chi: maximal allowed bond dimension;
    layers_per_regauging: how many layers are between consequent regauging of a network;
    trotter_steps: the total number of trotter steps in simulated dynamics;
    accuracy: an accuracy threshold used in convergence criterion for belief propagation;
    theta: an angle parameter in X rotation gate;
    is_infinite: a flag specifying the type of the model (127 qubits or infinite);
    synchronous_update: a flag specifying the schedule (synchronous or sequential);
    max_belief_propagation_iterations: a maximal allowed number of belief propagation iterations;
    key: jax random seed;
    traversal_type: the type of traversal (either 'dfs' standing for depth first search
        or 'bfs' standing for breadth first search).
Returns:
    a list with Z mean values versus time."""


def run_eagle_simulation(
    max_chi: int,
    layers_per_regauging: int,
    trotter_steps: int,
    accuracy: Union[float, Array],
    theta: float,
    is_infinite: bool = True,
    synchronous_update: bool = False,
    max_belief_propagation_iterations: Optional[int] = None,
    key: Array = PRNGKey(42),
    traversal_type: str = "dfs",
) -> List[Array]:

    # Eagle processor lattice initialization
    if is_infinite:
        eagle_lattice = get_heavy_hex_ibm_eagle_lattice_infinite()
    else:
        eagle_lattice = get_heavy_hex_ibm_eagle_lattice()

    # an iterator that traverses nodes and edges of a tensor graph
    traverser = list(eagle_lattice.get_traversal_iterator() or iter([]))

    # a target qubit id whose dynamics is necessary to compute
    if is_infinite:
        target_qubit_id = 2
    else:
        target_qubit_id = 62

    # an instance of belief propagation based quantum emulator
    quantum_emulator = BPQuantumEmulator(
        eagle_lattice,
        key,
        max_chi,
        eagle_lattice.edges_number * layers_per_regauging,
        accuracy,
        max_belief_propagation_iterations,
        synchronous_update,
        traversal_type,
    )

    # interaction gate
    interaction_gate = jnp.diag(
        jnp.exp(
            jnp.array(
                [
                    1j * jnp.pi / 4,
                    -1j * jnp.pi / 4,
                    -1j * jnp.pi / 4,
                    1j * jnp.pi / 4,
                ]
            )
        )
    ).reshape((2, 2, 2, 2))

    # X rotation gate
    mixing_gate = jnp.array(
        [
            jnp.cos(theta / 2),
            -1j * jnp.sin(theta / 2),
            -1j * jnp.sin(theta / 2),
            jnp.cos(theta / 2),
        ]
    ).reshape((2, 2))

    # simulation loop
    target_qubit_dynamics = []
    dens = quantum_emulator.dens_q1(target_qubit_id)
    log.info(f"Target qubit density matrix at layer 0: {dens}")
    target_qubit_dynamics.append((dens[0, 0] - dens[1, 1]).real)
    for layer_number in range(1, trotter_steps + 1):
        log.info(f"Layer number {layer_number} is running")
        # layer of mixing gates
        for node_id in range(eagle_lattice.nodes_number):
            quantum_emulator.apply_q1(mixing_gate, node_id)
        # layer of interaction gates
        for element in traverser:
            if isinstance(element, Edge):
                element_id = element.id
                assert isinstance(element_id, tuple)
                quantum_emulator.apply_q2(
                    interaction_gate,
                    element_id[0],
                    element_id[1],
                )
        # target qubit density matrix
        dens = quantum_emulator.dens_q1(target_qubit_id)
        log.info(f"Target qubit density matrix at layer {layer_number}: {dens}")
        log.info(f"Layer number {layer_number} is finished")
        target_qubit_dynamics.append((dens[0, 0] - dens[1, 1]).real)
    return target_qubit_dynamics


def main():

    # This set of parameters is for dynamics simulation of qubit from infinite Eagle processor
    # that repeats Fig. 4 e) from https://arxiv.org/abs/2306.14887.
    # It is already very difficult to simulate with MPS like approaches
    # but possible with belief propagation based tensor graphs.
    theta = 0.8
    max_chi = 200
    layers_per_regauging = 1
    trotter_steps = 20
    accuracy = 1e-5
    max_belief_propagation_iterations = None
    is_infinite = True

    # simulation
    target_qubit_dynamics = run_eagle_simulation(
        max_chi,
        layers_per_regauging,
        trotter_steps,
        accuracy,
        theta,
        is_infinite=is_infinite,
        max_belief_propagation_iterations=max_belief_propagation_iterations,
    )

    # plotting
    if is_infinite:
        qubits_number = "inf"
    else:
        qubits_number = 127
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(target_qubit_dynamics, "b")
    plot_info = f"""
    IBM Eagle quantum processor dynamics simulation.
    Configuration:
        qubits_number:        {qubits_number}
        theta:                {theta}
        trotter_steps:        {trotter_steps}
        layers_per_regauging: {layers_per_regauging}
        max_bond_dimension:   {max_chi}
    """
    fig.text(0.07, 0.9, plot_info)
    ax.set_ylabel("<Z>")
    ax.set_xlabel("Trotter steps")
    ax.set_ylim(bottom=0, top=1)
    fig.savefig(
        f"{os.path.dirname(os.path.realpath(__file__))}/ibm_infinite_eagle_dynamics.pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
