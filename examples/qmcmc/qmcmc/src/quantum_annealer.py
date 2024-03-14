from typing import Union, List, Dict, Hashable
import logging
import jax.numpy as jnp
from jax import Array
from .base_annealer import Annealer
from .energy_function import EnergyFunction
from .quantum_utils import (
    _tensor_graph_init,
    _compute_alpha,
    _init_q2_gates,
    _init_q1_gates,
)
from dequantizer import BPQuantumEmulator

log = logging.getLogger(__name__)


class QuantumAnnealer(Annealer):
    def __init__(
        self,
        energy_function: EnergyFunction,
        gamma: Union[float, Array],
        max_chi: int,
        layers_per_regauging: int,
        max_belief_propagation_iterations: int,
        time_step_per_layer: List[Union[Array, float]],
        key: Array,
        accuracy: Union[float, Array],
        synchronous_update: bool,
        traversal_type: str,
    ):
        super().__init__(energy_function)
        self.__tensor_graph = _tensor_graph_init(energy_function)
        self.__nodes_number = energy_function.fields.shape[0]
        self.__gamma = gamma
        self.__alpha = _compute_alpha(energy_function)
        self.__energy_function = energy_function
        self.__time_step_per_layer = time_step_per_layer
        self.__quantum_emulator = BPQuantumEmulator(
            self.__tensor_graph,
            key,
            max_chi,
            layers_per_regauging * self.__tensor_graph.edges_number,
            accuracy,
            max_belief_propagation_iterations,
            synchronous_update,
            traversal_type,
        )
        log.info(f"Alpha value: {self.__alpha}")

    def _transition(self, current_configuration: Array, _: Array) -> Array:
        self._init_state(current_configuration)
        for i, tau in enumerate(self.__time_step_per_layer):
            if i % 2 == 0:
                q1_gates = _init_q1_gates(self.__energy_function, tau, self.__gamma)
                assert (
                    len(q1_gates) == self.spins_number
                ), f"{len(q1_gates)} != {self.spins_number}"
                for _id, gate in enumerate(q1_gates):
                    self.__quantum_emulator.apply_q1(gate, _id)
            else:
                q2_gates = _init_q2_gates(
                    self.__energy_function,
                    self.__tensor_graph,
                    tau,
                    self.__alpha,
                    self.__gamma,
                )
                for (id1, id2), gate in q2_gates.items():
                    self.__quantum_emulator.apply_q2(gate, id1, id2)
        measurement_results = []
        for _id in range(self.__nodes_number):
            measurement_results.append(self.__quantum_emulator.measure(_id))
        new_config = 1 - 2 * jnp.array(measurement_results)
        log.info(f"New spins configuration: {new_config}")
        return new_config

    def _transition_probabilities_ratio(
        self,
        current_configuration: Array,
        potential_configuration: Array,
    ) -> Array:
        return jnp.array(1.0)

    def _init_state(self, config: Array):
        config_dict: Dict[Hashable, int] = {i: int(s) for i, s in enumerate(config)}
        self.__quantum_emulator.set_to_product_state(config_dict)
