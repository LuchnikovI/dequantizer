from typing import Dict
import jax.numpy as jnp
from jax import Array
from jax.random import split, uniform
from ..graph import TensorGraph, Node, NodeID
from .base_quantum_emulator import QuantumEmulator

"""An exact quantum computer emulator that serves for testing needs."""


class ExactQuantumEmulator(QuantumEmulator):
    """Initializes quantum emulator.
    Args:
        tensor_graph: a tensor graph object that specifies connectivity of
            the simulated processor;
        key: jax random seed."""

    def __init__(
        self,
        tensor_graph: TensorGraph,
        key: Array,
    ):
        super().__init__(tensor_graph, key)
        self.__qubits_number = tensor_graph.nodes_number
        node2pos = {}
        pos = 0
        for element in tensor_graph.get_traversal_iterator() or iter([]):
            if isinstance(element, Node):
                node2pos[element.id] = pos
                pos += 1
        self.__node2pos = node2pos
        state = jnp.zeros((2**self.__qubits_number,), dtype=jnp.complex128)
        self.__state = state.at[0].set(1.0)

    """Sets emulator state to a product state.
    Args:
        state: state of a qubit (0 or 1) per node."""

    def set_to_product_state(self, state: Dict[NodeID, int]):
        shift = 0
        for node_id, pos in self.__node2pos.items():
            if state[node_id] == 1:
                shift += 2 ** (self.__qubits_number - pos - 1)
        state_arr = jnp.zeros((2**self.__qubits_number,), dtype=jnp.complex128)
        self.__state = state_arr.at[shift].set(1.0)
        norm = jnp.linalg.norm(self.__state)
        assert jnp.isclose(norm, 1.0).all(), f"{norm}"

    """Applies a two-qubit gate.
    Args:
        gate: an array of shape (2, 2, 2, 2) representing a unitary gate,
            where for each index of this array we introduce a specific name
            inspired by CNOT gate semantic:
                an index number 0 we call controlling output,
                an index number 1 we call controlled output,
                an index number 2 we call controlling input,
                an index number 3 we call controlled input;
        controlling_node_id: a Node ID that is connected with the controlling input;
        controlled_node_id: a Node ID that is connected with controlled input."""

    def apply_q2(
        self,
        gate: Array,
        controlling_node_id: NodeID,
        controlled_node_id: NodeID,
    ):
        pos1 = self.__node2pos[controlling_node_id]
        pos2 = self.__node2pos[controlled_node_id]
        back_permutation = list(range(self.__qubits_number - 2))
        if pos1 < pos2:
            back_permutation.insert(pos1, self.__qubits_number - 2)
            back_permutation.insert(pos2, self.__qubits_number - 1)
        else:
            back_permutation.insert(pos2, self.__qubits_number - 1)
            back_permutation.insert(pos1, self.__qubits_number - 2)
        state = jnp.tensordot(
            self.__state.reshape(self.__qubits_number * (2,)),
            gate,
            axes=[[pos1, pos2], [2, 3]],
        )
        self.__state = jnp.transpose(state, back_permutation).reshape((-1,))

    """Applies a one-qubit gate.
    Args:
        gate: an array of shape (2, 2) representing a unitary gate,
            where zeroth index is the output and first index is the input;
        node_id: a Node ID where to apply a gate."""

    def apply_q1(self, gate: Array, node_id: NodeID):
        pos = self.__node2pos[node_id]
        back_permutation = (
            list(range(pos))
            + [self.__qubits_number - 1]
            + list(range(pos, self.__qubits_number - 1))
        )
        state = jnp.tensordot(
            self.__state.reshape(self.__qubits_number * (2,)), gate, axes=[[pos], [1]]
        )
        self.__state = jnp.transpose(state, back_permutation).reshape((-1,))

    """Computes density matrix of a specific node.
    Args:
        node_id: ID of a node.
    Returns:
        an array of shape (2, 2) representing a density matrix."""

    def dens_q1(self, node_id: NodeID) -> Array:
        pos = self.__node2pos[node_id]
        if pos != 0:
            state = (
                self.__state.reshape(self.__qubits_number * (2,))
                .swapaxes(axis1=0, axis2=pos)
                .reshape((2, -1))
            )
        else:
            state = self.__state.reshape((2, -1))
        dens = state @ state.T.conj()
        assert jnp.isclose(jnp.trace(dens), 1.0, 1e-8), f"{jnp.trace(dens)}"
        return dens

    """Measures a specific node.
    Args:
        node_id: ID of a node.
    Returns:
        an integer (0 or 1) representing the measurement results."""

    def measure(self, node_id: NodeID) -> int:
        dens = self.dens_q1(node_id)
        assert jnp.isclose(jnp.trace(dens), 1.0, 1e-8)
        key, subkey = split(self.prng_key)
        self.prng_key = key
        sample = uniform(subkey, (1,))
        p = dens[0, 0].real
        if p > sample:
            result = 0
            norm = p
        else:
            result = 1
            norm = 1 - p
        proj = jnp.zeros((2, 2))
        proj = proj.at[result, result].set(1.0)
        self.apply_q1(proj, node_id)
        self.__state /= jnp.sqrt(norm)
        # this is to keep seed consistent with BP based emulator
        key, subkey = split(self.prng_key)
        self.prng_key = key
        return result
