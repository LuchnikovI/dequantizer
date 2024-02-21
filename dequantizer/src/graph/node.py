from typing import List, Dict
from .element import Element, NodeID, EdgeID

"""Node class."""


class Node(Element):

    def __init__(self, dimension: int, node_id: NodeID):
        super().__init__(dimension, node_id)
        self.__bond_shape: tuple[int, ...] = ()
        self.__neighbors: List[Element] = []
        self.__neighbors2indices: Dict[EdgeID, int] = {}

    @property
    def degree(self) -> int:
        return len(self.__bond_shape)

    def _add_element(self, element: Element):
        self.__neighbors.append(element)
        element_id = element.id
        if not isinstance(element_id, tuple):
            raise NotImplementedError(
                "This branch is unreachable if the code is correct."
            )
        self.__neighbors2indices[element_id] = len(self.__neighbors) - 1
        self.__bond_shape = self.__bond_shape + (element.dimension,)

    @property
    def neighbors(self) -> List[Element]:
        return self.__neighbors

    """Gets the shape of the node tensor except the physical dimension."""

    @property
    def bond_shape(self) -> tuple[int, ...]:
        return self.__bond_shape

    @bond_shape.setter
    def bond_shape(self, value: tuple[int, ...]):
        self.__bond_shape = value

    """Gets a mapping from edge IDs to corresponding tensor indices numbers."""

    @property
    def indices(self) -> Dict[EdgeID, int]:
        return self.__neighbors2indices
