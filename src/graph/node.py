from typing import List
from .element import Element, NodeID
from jax import Array

"""Node class."""


class Node(Element):

    def __init__(self, dimension: int, id: NodeID):
        super().__init__(dimension, id)
        self.__bond_shape: tuple[int, ...] = ()
        self.__neighbors: List[Element] = []

    """Gets the shape of the node tensor except the physical dimension."""

    @property
    def bond_shape(self) -> tuple[int, ...]:
        return self.__bond_shape

    @property
    def degree(self) -> int:
        return len(self.__bond_shape)

    def _add_element(self, element: Element):
        self.__neighbors.append(element)
        self.__bond_shape = self.__bond_shape + (element.dimension,)

    @property
    def neighbors(self) -> List[Element]:
        return self.__neighbors
