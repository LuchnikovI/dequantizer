import os

os.environ["JAX_ENABLE_X64"] = "True"

from typing import Union

import jax

jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey, randint, split
from ..graph import Node, Edge, TensorGraph, get_random_tree_tensor_graph
from .bp_quantum_emulator import BPQuantumEmulator
from .exact_quantum_emulator import ExactQuantumEmulator
from .gate_generators import _get_random_q1_gate, _get_random_ising_like_q2_gate
from .graph_generators import _small_heavy_hex, _heavy_cube_graph


def _bp_quantum_emulator_test(
    tensor_graph: TensorGraph, layers_number: int, accuracy: Union[float, Array]
):
    key = PRNGKey(42)
    traverser = list(tensor_graph.get_traversal_iterator() or iter([]))
    key, subkey = split(key)
    exact_emulator = ExactQuantumEmulator(tensor_graph, subkey)
    bp_emulator = BPQuantumEmulator(
        tensor_graph,
        subkey,
        1024,
        tensor_graph.edges_number,
        belief_propagation_accuracy=1e-6,
    )
    key, subkey = split(key)
    random_product_state_arr = randint(
        subkey, (tensor_graph.nodes_number,), minval=0, maxval=2
    )
    random_product_state = {
        node.id: int(random_product_state_arr[i])
        for i, node in enumerate(filter(lambda x: isinstance(x, Node), traverser))
    }
    exact_emulator.set_to_product_state(random_product_state)
    bp_emulator.set_to_product_state(random_product_state)
    # check correctness of the initial state
    for node_id, state in random_product_state.items():
        bp_dens = bp_emulator.dens_q1(node_id)
        exact_dens = exact_emulator.dens_q1(node_id)
        correct_dens = jnp.zeros((2, 2))
        correct_dens = correct_dens.at[state, state].set(1.0)
        assert jnp.isclose(
            bp_dens, correct_dens, 1e-8
        ).all(), f"{bp_dens}, {correct_dens}"
        assert jnp.isclose(
            exact_dens, correct_dens, 1e-8
        ).all(), f"{exact_dens}, {correct_dens}"
        key, subkey = split(key)
        measurement_result = bp_emulator.measure(node_id)
        assert measurement_result == state
    # apply 2 layers of gates
    for _ in range(layers_number):
        for element in traverser:
            if isinstance(element, Node):
                key, subkey = split(key)
                q1_gate = _get_random_q1_gate(subkey)
                exact_emulator.apply_q1(q1_gate, element.id)
                bp_emulator.apply_q1(q1_gate, element.id)
            elif isinstance(element, Edge):
                element_id = element.id
                if not isinstance(element_id, tuple):
                    raise ValueError("Edge ID must be tuple, this is a bug.")
                key, subkey = split(key)
                q2_gate = _get_random_ising_like_q2_gate(subkey)
                exact_emulator.apply_q2(q2_gate, element_id[0], element_id[1])
                bp_emulator.apply_q2(q2_gate, element_id[0], element_id[1])
    # check correctness of density matrices
    for element in traverser:
        if isinstance(element, Node):
            exact_dens = exact_emulator.dens_q1(element.id)
            bp_dens = bp_emulator.dens_q1(element.id)
            assert (
                jnp.linalg.norm(exact_dens - bp_dens) < accuracy
            ), f"{exact_dens}, {bp_dens}"
    # check correctness of measurements
    exact_emulator.prng_key = bp_emulator.prng_key
    for element in traverser:
        if isinstance(element, Node):
            exact_dens = exact_emulator.dens_q1(element.id)
            bp_dens = bp_emulator.dens_q1(element.id)
            assert (
                jnp.linalg.norm(exact_dens - bp_dens) < accuracy
            ), f"{exact_dens}, {bp_dens}"
            exact_measurement = exact_emulator.measure(element.id)
            bp_measurement = bp_emulator.measure(element.id)
            assert exact_measurement == bp_measurement


def test_bp_quantum_emulator():
    key = PRNGKey(42)
    tensor_graph = _heavy_cube_graph()
    _bp_quantum_emulator_test(tensor_graph, 2, 3e-3)
    print("Cube graph: OK")
    tensor_graph = _small_heavy_hex()
    _bp_quantum_emulator_test(tensor_graph, 4, 1e-4)
    print("Small heavy hex: OK")
    tensor_graph = get_random_tree_tensor_graph(15, 2, 14 * [1], key)
    print("Random tree: OK")
    _bp_quantum_emulator_test(tensor_graph, 4, 1e-4)
