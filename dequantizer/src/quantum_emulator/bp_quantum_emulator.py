from typing import Union, Dict, Optional
import logging
import jax.numpy as jnp
from jax import Array, jit
from jax.random import split, uniform
from ..graph import (
    TensorGraph,
    NodeID,
    EdgeID,
    Node,
    get_message_random_nonnegative_initializer,
    get_tensor_std_state_initializer,
)
from ..mappings import (
    get_belief_propagation_map,
    get_vidal_gauge_fixing_map,
    get_vidal_gauge_distance_map,
    get_symmetric_gauge_fixing_map,
    lambdas2messages,
    get_one_side_density_matrix,
)
from .base_quantum_emulator import QuantumEmulator

log = logging.getLogger(__name__)

"""An approximate quantum computer emulator on a specific interaction graph.
Underneath it uses Belief Propagation for approximate gauging and inference
on an arbitrary tensor graph."""


class BPQuantumEmulator(QuantumEmulator):
    """Initializes quantum emulator.
    Args:
        tensor_graph: a tensor graph object that specifies connectivity of
            the underlying tensor network;
        key: jax random seed;
        max_chi: maximal bond dimension allowed in the underlying
            tensor network;
        gates_number_per_regauging: number of gates run between consequent
            regauging;
        belief_propagation_accuracy: an accuracy used in belief propagation
            convergence criterion;
        max_belief_propagation_iterations: a maximal number of iterations
            allowed for belief propagation;
        synchronous_update: a flag specifying if synchronous updated used
            in belief propagation procedure;
        traversal_type: a type of graph traversal ("dfs" or "bfs")."""

    def __init__(
        self,
        tensor_graph: TensorGraph,
        key: Array,
        max_chi: int,
        gates_number_per_regauging: int,
        belief_propagation_accuracy: Union[float, Array] = 1e-5,
        max_belief_propagation_iterations: Optional[int] = 100,
        synchronous_update: bool = False,
        traversal_type: str = "dfs",
    ):
        super().__init__(tensor_graph, key)
        log.info(
            f"""
            An instance of Belief-Propagation based emulator has been created, parameters:
                jax_random_seed: {key}
                max_chi: {max_chi}
                gates_number_per_regauging: {gates_number_per_regauging}
                belief_propagation_accuracy: {belief_propagation_accuracy}
                synchronous_update: {synchronous_update}
                traversal_type: {traversal_type}
            """
        )
        self.__max_chi = max_chi
        self.__max_belief_propagation_iterations = max_belief_propagation_iterations
        self.__gates_number_per_regauging = gates_number_per_regauging
        self.__belief_propagation_accuracy = belief_propagation_accuracy
        self.__gates_num_passed_after_regauging = 0
        self.__traverser = list(
            tensor_graph.get_traversal_iterator(ordering=traversal_type) or iter([])
        )
        self.__bp_map = jit(
            get_belief_propagation_map(self.__traverser, synchronous_update)
        )
        self.__vg_map = jit(get_vidal_gauge_fixing_map(self.__traverser, 1e-10))
        self.__vd_map = jit(get_vidal_gauge_distance_map(self.__traverser, 1e-10))
        self.__sg_map = jit(get_symmetric_gauge_fixing_map(self.__traverser, 1e-10))
        self.__lambdas2messages = jit(lambdas2messages)
        self.__tensors = self.tensor_graph.init_tensors(
            get_tensor_std_state_initializer()
        )
        self.__lambdas: Optional[Dict[EdgeID, Array]] = None
        self._set_to_vidal_canonical()

    """Sets emulator state to a product state.
    Args:
        state: state of a qubit (0 or 1) per node."""

    def set_to_product_state(self, state: Dict[NodeID, int]):
        self.tensor_graph.to_product()

        def _initializer(node: Node) -> Array:
            s = state[node.id]
            tensor = jnp.zeros((2,))
            tensor = tensor.at[s].set(1.0)
            return tensor.reshape(node.bond_shape + (node.dimension,))

        self.__tensors = self.tensor_graph.init_tensors(_initializer)
        self._set_to_vidal_canonical()
        self.__gates_num_passed_after_regauging = 0

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
        if self.__lambdas is None:
            raise ValueError("Lambdas are not defined, more likely it is bug.")
        if self.__gates_num_passed_after_regauging >= self.__gates_number_per_regauging:
            self._vidal_regauging()
            self.__gates_num_passed_after_regauging = 0
        self.tensor_graph.apply2_to_vidal_canonical(
            controlling_node_id,
            controlled_node_id,
            self.__tensors,
            self.__lambdas,
            gate,
            1e-10,
            None,
            self.__max_chi,
        )
        self.__gates_num_passed_after_regauging += 1

    """Applies a one-qubit gate.
    Args:
        gate: an array of shape (2, 2) representing a unitary gate,
            where zeroth index is the output and first index is the input;
        node_id: a Node ID where to apply a gate."""

    def apply_q1(self, gate: Array, node_id: NodeID):
        self.__tensors = self.tensor_graph.apply1(
            node_id,
            self.__tensors,
            gate,
        )

    """Computes density matrix of a specific node.
    Args:
        node_id: ID of a node.
    Returns:
        an array of shape (2, 2) representing a density matrix."""

    def dens_q1(self, node_id: NodeID) -> Array:
        if self.__gates_num_passed_after_regauging != 0:
            self._vidal_regauging()
            self.__gates_num_passed_after_regauging = 0
        if self.__lambdas is None:
            raise ValueError("This brunch is unreachable if the code is correct")
        tensors = self.__sg_map(self.__tensors, self.__lambdas)
        messages = self.__lambdas2messages(self.__lambdas)
        node = self.tensor_graph.get_node(node_id)
        if node is None:
            raise ValueError(
                f"There is no node with id {node_id}, more likely it is a bug."
            )
        dens = get_one_side_density_matrix(
            node,
            tensors,
            messages,
        )
        assert jnp.isclose(jnp.trace(dens), 1.0, 1e-8), jnp.trace(dens)
        return dens

    """Measures a specific node.
    Args:
        node_id: ID of a node.
    Returns:
        an integer (0 or 1) representing the measurement results."""

    def measure(self, node_id: NodeID) -> int:
        dens = self.dens_q1(node_id)
        key, subkey = split(self.prng_key)
        self.prng_key = key
        sample = uniform(subkey, (1,))
        p = dens[0, 0].real
        if p > sample:
            result = 0
        else:
            result = 1
        proj = jnp.zeros((2, 2))
        proj = proj.at[result, result].set(1.0)
        self.apply_q1(proj, node_id)
        if self.__lambdas is None:
            raise ValueError()
        self.__tensors = self.__sg_map(self.__tensors, self.__lambdas)
        self._set_to_vidal_canonical()
        self.__gates_num_passed_after_regauging = 0
        return result

    # --------------------------------------------------------------------------

    def _set_to_vidal_canonical(self):
        log.info("Setting state to the Vidal canonical form started")
        key, subkey = split(self.prng_key)
        self.prng_key = key
        message_initializer = get_message_random_nonnegative_initializer(subkey)
        messages = self.tensor_graph.init_messages(message_initializer)
        vidal_dist = jnp.finfo(jnp.float64).max
        iters = 0
        while vidal_dist > self.__belief_propagation_accuracy:
            if (
                self.__max_belief_propagation_iterations is not None
                and self.__max_belief_propagation_iterations <= iters
            ):
                log.warning(
                    "Belief propagation reached the maximal number of iterations."
                )
                break
            messages = self.__bp_map(self.__tensors, messages)
            tensors, lambdas = self.__vg_map(self.__tensors, messages)
            vidal_dist = self.__vd_map(tensors, lambdas)
            log.info(f"Vidal distance: {vidal_dist}")
            iters += 1
        log.info("State is set to the Vidal canonical form")
        self.__lambdas = lambdas
        self.__tensors = tensors

    def _vidal_regauging(self):
        log.info("Vidal regauging started")
        if self.__lambdas is None:
            raise ValueError("Lambdas are not defined, more likely it is bug.")
        tensors = self.__sg_map(self.__tensors, self.__lambdas)
        messages = self.__lambdas2messages(self.__lambdas)
        vidal_dist = jnp.finfo(jnp.float64).max
        iters = 0
        while vidal_dist > self.__belief_propagation_accuracy:
            if (
                self.__max_belief_propagation_iterations is not None
                and self.__max_belief_propagation_iterations <= iters
            ):
                log.warning(
                    "Belief propagation reached the maximal number of iterations."
                )
                break
            messages = self.__bp_map(tensors, messages)
            tensors, lambdas = self.__vg_map(tensors, messages)
            vidal_dist = self.__vd_map(tensors, lambdas)
            log.info(f"Vidal distance: {vidal_dist}")
            iters += 1
        log.info("Regauging finished")
        self.__lambdas = lambdas
        self.__tensors = tensors
