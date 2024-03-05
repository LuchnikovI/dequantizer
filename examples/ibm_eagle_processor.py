#!/usr/bin/env python3

"""In this example we simulate dynamics of 62-nd qubit (middle one) in Eagle
IBM quantum processor either for 127 qubits (the original IBM processor) 
or for its infinite modification.

Note, that simulation for an infinite processor is way faster due to 
the translational symmetry."""

import os

os.environ["JAX_ENABLE_X64"] = "True"
import jax

jax.config.update("jax_platform_name", "cpu")
import matplotlib

matplotlib.rcParams["font.family"] = "monospace"
from typing import Union, Dict, Tuple, List
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jit, Array
from jax.random import PRNGKey, split
from dequantizer import (
    get_heavy_hex_ibm_eagle_lattice_infinite,
    get_heavy_hex_ibm_eagle_lattice,
    get_belief_propagation_map,
    get_vidal_gauge_fixing_map,
    get_vidal_gauge_distance_map,
    get_symmetric_gauge_fixing_map,
    get_message_random_nonnegative_initializer,
    get_tensor_std_state_initializer,
    get_one_side_density_matrix,
    lambdas2messages,
    Node,
    Edge,
    NodeID,
    EdgeID,
)

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
    key: Array = PRNGKey(42),
    traversal_type: str = "dfs",
) -> List[Array]:

    # Eagle processor lattice initialization
    if is_infinite:
        eagle_lattice = get_heavy_hex_ibm_eagle_lattice_infinite()
    else:
        eagle_lattice = get_heavy_hex_ibm_eagle_lattice()

    # defining the tensor graph traverser
    traverser = list(
        eagle_lattice.get_traversal_iterator(ordering=traversal_type) or iter([])
    )
    # a map that performs a single belief propagation iteration
    bp_map = jit(get_belief_propagation_map(traverser, synchronous_update))

    # a map that fixes the Vidal gauge given the belief propagation is converged
    # (1e-10 constant is just a small value necessary for regularization of
    # some inversions)
    vg_map = jit(get_vidal_gauge_fixing_map(traverser, 1e-10))

    # a map that computes the distance to the Vidal gauge
    # (1e-10 constant is just a small value necessary to distinguish signal from noise)
    vd_map = jit(get_vidal_gauge_distance_map(traverser, 1e-10))

    # a map that fixes symmetric gauge
    sg_map = jit(get_symmetric_gauge_fixing_map(traverser, 1e-10))

    """This function performs belief propagation iterations until convergence
    and sets the tensor graph to the Vidal canonical form given the belief propagation
    is converged.
    Args:
        tensors: tensors forming a tensor graph;
        key: jax random seed.
    Returns:
        fixed gauge tensors and diagonal matrices sitting on edges (singular values)."""

    def set_to_vidal_gauge(
        tensors: Dict[NodeID, Array],
        key: Array,
    ) -> Tuple[Dict[NodeID, Array], Dict[EdgeID, Array]]:
        print("""Setting a tensor graph to the Vidal gauge...""")
        message_initializer = get_message_random_nonnegative_initializer(key)
        messages = eagle_lattice.init_messages(message_initializer)
        vidal_dist = jnp.finfo(jnp.float64).max
        while vidal_dist > accuracy:
            messages = bp_map(tensors, messages)
            canonical_tensors, lmbds = vg_map(tensors, messages)
            vidal_dist = vd_map(canonical_tensors, lmbds)
            print(f"\tVidal distance: {vidal_dist}")
        print("""Vidal gauge is set.""")
        return canonical_tensors, lmbds

    """This function performs regauging of the tensor graph and compute
    the value of the target observable (<Z> of the 62-nd qubit)
    by running belief propagation.
    Args:
        tensors: fixed gauge tensors that need to be corrected;
        lambdas: diagonal matrices sitting on edges (singular values)
            that need to be corrected.
    Returns:
        corrected (regauged) fixed gauge tensors and diagonal matrices
        sitting on edges."""

    def vidal_regauging_and_computing_observables(
        tensors: Dict[NodeID, Array],
        lambdas: Dict[EdgeID, Array],
    ) -> Tuple[Tuple[Dict[NodeID, Array], Dict[EdgeID, Array]], Array]:
        print("Running regauging and observable evaluation...")
        tensors = sg_map(tensors, lambdas)
        messages = lambdas2messages(lambdas)
        vidal_dist = jnp.finfo(jnp.float64).max
        while vidal_dist > accuracy:
            messages = bp_map(tensors, messages)
            canonical_tensors, lmbds = vg_map(tensors, messages)
            vidal_dist = vd_map(canonical_tensors, lmbds)
            print(f"\tVidal distance: {vidal_dist}")
        print("Regauging finished.")
        if is_infinite:
            target_node = eagle_lattice.get_node(2)
        else:
            target_node = eagle_lattice.get_node(62)
        assert target_node is not None  # node must be present in a graph
        dens = get_one_side_density_matrix(target_node, tensors, messages)
        print("Observable is computed.")
        return (canonical_tensors, lmbds), (dens[0, 0] - dens[1, 1]).real

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

    # tensors initialization
    tensors = eagle_lattice.init_tensors(get_tensor_std_state_initializer())

    # setting tensor graph to the Vidal gauge
    key, subkey = split(key)
    tensors, lambdas = set_to_vidal_gauge(tensors, subkey)

    # simulation loop
    target_qubit_dynamics = []
    for layer_number in range(trotter_steps):

        # periodic regauging each `layers_per_regauging` layers
        if layer_number % layers_per_regauging == 0:
            (tensors, lambdas), z62 = vidal_regauging_and_computing_observables(
                tensors, lambdas
            )
            target_qubit_dynamics.append(z62)

        print(f"Running layer number {layer_number}...")

        # X rotation gate application
        for element in traverser:
            if isinstance(element, Node):
                tensors = eagle_lattice.apply1(element.id, tensors, mixing_gate)

        # Interaction gate application
        for element in traverser:
            if isinstance(element, Edge):
                element_id = element.id
                assert isinstance(element_id, tuple)
                tensors, lambdas = eagle_lattice.apply2_to_vidal_canonical(
                    element_id[0],
                    element_id[1],
                    tensors,
                    lambdas,
                    interaction_gate,
                    1e-10,  # this is a small constant to regularize some inversions
                    None,
                    max_chi,
                )
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
    is_infinite = True

    # simulation
    target_qubit_dynamics = run_eagle_simulation(
        max_chi,
        layers_per_regauging,
        trotter_steps,
        accuracy,
        theta,
        is_infinite=is_infinite,
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
