from typing import List
from .element import Element, EdgeID

"""Edge class."""


class Edge(Element):

    def __init__(self, dimension: int, edge_id: EdgeID):
        super().__init__(dimension, edge_id)
        self.__neighbors: List[Element] = []

    def _add_element(self, node: Element):
        self.__neighbors.append(node)

    @property
    def degree(self) -> int:
        return len(self.__neighbors)

    @property
    def neighbors(self) -> List[Element]:
        return self.__neighbors
