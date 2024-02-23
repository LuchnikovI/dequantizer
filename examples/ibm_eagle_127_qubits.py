#!/usr/bin/env python3

import os

os.environ["JAX_ENABLE_X64"] = "True"
# Fixing some compilation flags of XLA is necessary to simplify
# the compilation process, otherwise it is prohibitively slow and memory demanding
os.environ["XLA_FLAGS"] = (
    "--xla_backend_optimization_level=0 --xla_llvm_disable_expensive_passes=true"
)

import jax
jax.config.update('jax_platform_name', 'cpu')
import matplotlib

matplotlib.rcParams["font.family"] = "monospace"
from typing import Union, Dict, Tuple, List
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jit, Array
from jax.random import PRNGKey, split
from dequantizer import (
    get_heavy_hex_ibm_eagle_lattice_infinite,
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

def run_eagle_simulation(
    max_chi: int,
    layers_per_regauging: int,
    layers_number: int,
    accuracy: Union[float, Array],
    theta: float,
    synchronous_update: bool = False,
    key: Array = PRNGKey(42),
    traversal_type: str = "dfs",
) -> List[Array]:
    eagle_lattice = get_heavy_hex_ibm_eagle_lattice_infinite()
    traverser = list(
        eagle_lattice.get_traversal_iterator(ordering=traversal_type) or iter([])
    )
    bp_map = jit(get_belief_propagation_map(traverser, synchronous_update))
    vg_map = jit(get_vidal_gauge_fixing_map(traverser, 1e-10))
    vd_map = jit(get_vidal_gauge_distance_map(traverser))
    sg_map = jit(get_symmetric_gauge_fixing_map(traverser))
    def _set_to_vidal_gauge(
        tensors: Dict[NodeID, Array],
        key: Array,
    ) -> Tuple[Dict[NodeID, Array], Dict[EdgeID, Array]]:
        message_initializer = get_message_random_nonnegative_initializer(key)
        messages = eagle_lattice.init_messages(message_initializer)
        vidal_dist = jnp.finfo(jnp.float64).max
        while vidal_dist > accuracy:
            messages = bp_map(tensors, messages)
            canonical_tensors, lmbds = vg_map(tensors, messages)
            vidal_dist = vd_map(canonical_tensors, lmbds)
        return canonical_tensors, lmbds
    def _vidal_regauging_and_computing_observables(
        tensors: Dict[NodeID, Array],
        lambdas: Dict[EdgeID, Array],
    ) -> Tuple[Tuple[Dict[NodeID, Array], Dict[EdgeID, Array]], Array]:
        print("Running regauging...")
        tensors = sg_map(tensors, lambdas)
        messages = lambdas2messages(lambdas)
        vidal_dist = jnp.finfo(jnp.float64).max
        while vidal_dist > accuracy:
            messages = bp_map(tensors, messages)
            canonical_tensors, lmbds = vg_map(tensors, messages)
            vidal_dist = vd_map(canonical_tensors, lmbds)
            print(f"\tVidal distance: {vidal_dist}")
        print("Regauging finished.")
        dens = get_one_side_density_matrix(eagle_lattice.get_node(2), tensors, messages)
        return (canonical_tensors, lmbds), (dens[0, 0] - dens[1, 1]).real
    interaction_gate = jnp.diag(jnp.exp(jnp.array([
        1j * jnp.pi / 4, -1j * jnp.pi / 4, -1j * jnp.pi / 4, 1j * jnp.pi / 4
    ]))).reshape((2, 2, 2, 2))
    mixing_gate = jnp.array([
        jnp.cos(theta / 2), -1j * jnp.sin(theta / 2),
        -1j * jnp.sin(theta / 2), jnp.cos(theta / 2)
    ]).reshape((2, 2))
    tensors = eagle_lattice.init_tensors(get_tensor_std_state_initializer())
    key, subkey = split(key)
    tensors, lambdas = _set_to_vidal_gauge(tensors, subkey)
    z62_dynamics = []
    for layer_number in range(layers_number):
        if layer_number % layers_per_regauging == 0:
            (tensors, lambdas), z62 = _vidal_regauging_and_computing_observables(tensors, lambdas)
            z62_dynamics.append(z62)
        print(f"Running layer number {layer_number}...")
        for element in traverser:
            if isinstance(element, Node):
                tensors = eagle_lattice.apply1(element.id, tensors, mixing_gate)
        for element in traverser:
            if isinstance(element, Edge):
                tensors, lambdas = eagle_lattice.apply2_to_vidal_canonical(
                    element.id[0],
                    element.id[1],
                    tensors,
                    lambdas,
                    interaction_gate,
                    1e-10,
                    None,
                    max_chi,
                )
    return z62_dynamics
        

def main():
    key = PRNGKey(42)
    theta = 0.6
    max_chi = 128
    layers_per_regauging = 1
    layers_number = 2
    
    accuracy = 1e-5
    z62_dynamics = run_eagle_simulation(
        max_chi,
        layers_per_regauging,
        layers_number,
        accuracy,
        theta,
    )
    plt.figure(figsize=(10, 5))
    plt.plot(z62_dynamics, "b")
    plt.ylabel("<Z>")
    plt.xlabel("Trotter steps")
    plt.ylim(bottom=0, top=1)
    plt.savefig("ibm_infinite_eagle_dynamics.pdf")
        
if __name__ == "__main__":
    main()