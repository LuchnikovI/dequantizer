from typing import Dict
from abc import ABC, abstractmethod
from jax import Array
from ..graph import TensorGraph, NodeID


class QuantumEmulator(ABC):

    def __init__(
        self,
        tensor_graph: TensorGraph,
        key: Array,
    ):
        self.__tensor_graph = tensor_graph
        self.__key = key

    @property
    def prng_key(self) -> Array:
        return self.__key

    @prng_key.setter
    def prng_key(self, key: Array):
        self.__key = key

    @property
    def tensor_graph(self) -> TensorGraph:
        return self.__tensor_graph

    @tensor_graph.setter
    def tensor_graph(self, tensor_graph: TensorGraph):
        self.__tensor_graph = tensor_graph

    @abstractmethod
    def set_to_product_state(
        self,
        state: Dict[NodeID, int],
    ):
        pass

    @abstractmethod
    def apply_q2(
        self,
        gate: Array,
        controlling_node_id: NodeID,
        controlled_node_id: NodeID,
    ):
        pass

    @abstractmethod
    def apply_q1(
        self,
        gate: Array,
        node_id: NodeID,
    ):
        pass

    @abstractmethod
    def dens_q1(
        self,
        node_id: NodeID,
    ) -> Array:
        pass

    @abstractmethod
    def measure(
        self,
        node_id: NodeID,
    ) -> int:
        pass
