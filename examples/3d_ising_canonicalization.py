#!/usr/bin/env python3

"""In this example we build a 3d PEPS-like tensor network,
that can be seen as a square root of a classical 3D Ising model
partition function, i.e. the contraction of the network with
itself gives the value of the partition function. We bring it
to the Vidal canonical form and calculate average magnetization
for different inverse temperatures. This is an example from the paper
https://arxiv.org/pdf/2306.17837 (see the last paragraph of section 3.2).
This script saves resulting plots in the script directory."""

# !!! Fist iteration of BP could be very long due to jit compilation
# !!! remaining iterations are much faster

import os

os.environ["JAX_ENABLE_X64"] = "True"
# Fixing some compilation flags of XLA is necessary to simplify
# the compilation process, otherwise it is prohibitively slow and memory demanding
os.environ["XLA_FLAGS"] = (
    "--xla_backend_optimization_level=0 --xla_llvm_disable_expensive_passes=true"
)

import jax
jax.config.update("jax_platform_name", "cpu")

from dataclasses import dataclass
from typing import Union, Dict, List
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.family"] = "monospace"
import jax.numpy as jnp
from jax import Array, jit
from jax.random import PRNGKey
from dequantizer import (
    get_nd_lattice,
    get_potts_initializer,
    get_message_random_nonnegative_initializer,
    Node,
    NodeID,
    MessageID,
    get_belief_propagation_map,
    get_vidal_gauge_fixing_map,
    get_vidal_gauge_distance_map,
    get_one_side_density_matrix,
)

"""This class contains results of canonicalization,
average magnetization, number of iterations passed
before belief propagation convergence and the evolution of
the distance to the vidal gauge with number of iterations
for each inverse temperature."""


@dataclass
class CanonicalizationResult:
    tensors: List[Dict[NodeID, Array]]
    messages: List[Dict[MessageID, Array]]
    vidal_dist_per_iter: List[Array]
    number_of_iterations: List[int]
    average_magnetization: List[Array]


"""This function performs numerical experiments.
Args:
    lattice_dim: dimension of the cubic lattice (
        typically 1, 2 or 3);
    lattice_size: the edge size of the cubic lattice;
    betas: an array with inverse temperatures to experiment with;
    couplings: the amplitude of couplings between spins;
    field: the amplitude of local magnetic fields per spin;
    accuracy: an accuracy threshold used in convergence criterion.
        and some other approximate procedures;
    open_boundary: a flag specifying if open boundary conditions are used
        or periodic;
    synchronous_update: a flag specifying if synchronous update strategy is
        used or sequential one;
    key: jax random seed;
    traversal_type: a string specifying a type of tensor graph traversal (
        either 'dfs' standing for depth first search or 'bfs' standing for
        breadth first search).
Returns:
    CanonicalizationResult class (see above) with results.
    """


def process_ising_sqrt_partition_network(
    lattice_dim: int,
    lattice_size: int,
    betas: Array,
    coupling: Union[float, Array],
    field: Union[float, Array],
    accuracy: Union[float, Array],
    open_boundary: bool,
    synchronous_update: bool = False,
    key: Array = PRNGKey(42),
    traversal_type: str = "dfs",
) -> CanonicalizationResult:
    result = CanonicalizationResult([], [], [], [], [])
    # building a square root partition function networks
    ising_lattice = get_nd_lattice(lattice_dim * [lattice_size], 2, 2, open_boundary)
    # defining the traverser that specify how belief propagation iterates over network elements
    traverser = list(
        ising_lattice.get_traversal_iterator(ordering=traversal_type) or iter([])
    )
    # a map that performs Belief Propagation single iteration
    bp_map = jit(get_belief_propagation_map(traverser, synchronous_update))
    # a map that fixes Vidal gauge having converged messages after Belief Propagation
    vg_map = jit(get_vidal_gauge_fixing_map(traverser, jnp.array(accuracy)))
    # a map that computes the distance to the Vidal gauge
    vd_map = jit(get_vidal_gauge_distance_map(traverser, jnp.array(accuracy)))
    # loop over values of inverse temperature
    for i, beta in enumerate(betas):
        print(f"An iteration #{i} is started, inverse temperature is {beta}.")
        # an interaction factor
        two_spin_factor = jnp.exp(
            -coupling
            * beta
            * jnp.tensordot(jnp.array([1.0, -1.0]), jnp.array([1.0, -1.0]), axes=0)
        )
        # a local magnetic field factor
        one_spin_factor = jnp.exp(-field * beta * jnp.array([1.0, -1.0]))
        # an initializer that initializes node tensors in a way that the resulting
        # tensor network represents square root of the Ising model partition function
        tensor_initializer = get_potts_initializer(two_spin_factor, one_spin_factor)
        # an initializer of messages that sample random non-negative messages
        message_initializer = get_message_random_nonnegative_initializer(key)
        # messages
        messages = ising_lattice.init_messages(message_initializer)
        # node tensors
        tensors = ising_lattice.init_tensors(tensor_initializer)
        vidal_dist = jnp.finfo(jnp.float64).max
        vidal_dist_per_iter: List[Array] = []
        # Belief Propagation until convergence
        while vidal_dist > accuracy:
            # a single iteration of belief propagation
            new_messages = bp_map(tensors, messages)
            messages = new_messages
            canonical_tensors, lmbds = vg_map(tensors, messages)
            # vidal distance after an iteration
            vidal_dist = vd_map(canonical_tensors, lmbds)
            print(f"\tVidal distance: {vidal_dist}")
            vidal_dist_per_iter.append(vidal_dist)
        # computing average magnetization across the Ising model
        average_magnetization = jnp.array(0.0)
        for element in traverser:
            if isinstance(element, Node):
                dens = get_one_side_density_matrix(element, tensors, messages)
                average_magnetization += 2 * dens[0, 0].real - 1
        average_magnetization /= ising_lattice.nodes_number
        print(f"Iteration number {i} is finished, inverse temperature {beta}.")
        # Gathering results
        result.tensors.append(tensors)
        result.messages.append(messages)
        result.vidal_dist_per_iter.append(jnp.array(vidal_dist_per_iter))
        result.number_of_iterations.append(len(vidal_dist_per_iter))
        result.average_magnetization.append(average_magnetization)
    return result


def main():
    # experiment parameters
    lattice_dim = 3
    lattice_size = 10
    coupling = -1.0
    field = 0.5
    accuracy = 1e-8
    betas = jnp.linspace(0, 0.5, 100)
    open_boundary = True
    synchronous_update = False
    # experiment execution
    result = process_ising_sqrt_partition_network(
        lattice_dim,
        lattice_size,
        betas,
        coupling,
        field,
        accuracy,
        open_boundary,
        synchronous_update,
    )

    # plotting part

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(betas, result.number_of_iterations, "b")
    plot_info = f"""
    {lattice_dim}-D classical Ising model
    BP iterations number until convergence
    to the Vidal gauge with given accuracy.
    Configuration:
        linear_size:    {lattice_size}
        coupling:       {coupling}
        magnetic_field: {field}
        accuracy:       {accuracy}
        open_bc?:       {open_boundary}
    """
    fig.text(0.07, 0.9, plot_info)
    ax.set_ylabel("Iterations number")
    ax.set_xlabel("Inverse temperature")
    fig.savefig(
        f"{os.path.dirname(os.path.realpath(__file__))}/ising_iternum_vs_beta.pdf",
        bbox_inches="tight",
    )

    exp_number = len(betas) // 2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(result.vidal_dist_per_iter[exp_number], "b")
    ax.set_yscale("log")
    plot_info = f"""
    {lattice_dim}-D classical Ising model
    Distance to the Vidal gauge versus number
    of BP iteration.
    Configuration:
        inverse_temperature: {betas[exp_number]}
        linear_size:         {lattice_size}
        coupling:            {coupling}
        magnetic_field:      {field}
        accuracy:            {accuracy}
        open_bc?:            {open_boundary}
    """
    fig.text(0.07, 0.9, plot_info)
    ax.set_ylabel("Distance to the Vidal gauge")
    ax.set_xlabel("Iteration number")
    plt.savefig(
        f"{os.path.dirname(os.path.realpath(__file__))}/ising_vidaldist_vs_iter.pdf",
        bbox_inches="tight",
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(betas, result.average_magnetization, "b")
    plot_info = f"""
    {lattice_dim}-D classical Ising model
    Total average magnetization versus inverse temperature.
    Configuration:
        linear_size:    {lattice_size}
        coupling:       {coupling}
        magnetic_field: {field}
        accuracy:       {accuracy}
        open_bc?:       {open_boundary}
    """
    fig.text(0.07, 0.9, plot_info)
    ax.set_ylabel("Total average magnetization")
    ax.set_xlabel("Inverse temperature")
    fig.savefig(
        f"{os.path.dirname(os.path.realpath(__file__))}/ising_totalmagnetization_vs_beta.pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
