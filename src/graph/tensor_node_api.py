from typing import List
from abc import ABC, abstractmethod
from tensor_graph_api import EdgeID

"""Tensor node edge class."""
class Node(ABC):
    """Gets the physical dimension of the node."""

    @abstractmethod
    @property
    def physical_dimension(self) -> int:
        pass

    """Gets the mapping from the node associated tensor indices to the edge IDs.
    The order of IDs in the list corresponds to the order
    of indices."""

    @abstractmethod
    @property
    def indices2edges(self) -> List[EdgeID]:
        pass

    """Gets the shape of the node associated tensor."""

    @abstractmethod
    @property
    def shape(self) -> tuple[int, ...]:
        pass

    """Gets the rank (indices number) of the associated tensor."""

    @abstractmethod
    @property
    def rank(self) -> int:
        pass
