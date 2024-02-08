from typing import List
from abc import ABC, abstractmethod
from tensor_graph_api import NodeID

"""Tensor graph edge class."""
class Edge(ABC):
    """Gets the dimension of the edge."""

    @abstractmethod
    @property
    def dimension(self) -> int:
        pass

    """Gets the neighboring nodes list."""

    @abstractmethod
    @property
    def neighboring_nodes(self) -> List[NodeID]:
        pass
