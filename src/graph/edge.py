from typing import List
from element import Element, EdgeID

"""Edge class."""


class Edge(Element):

    def __init__(self, dimension: int, id: EdgeID):
        super().__init__(dimension, id)
        self.__neighbors: List[Element] = []

    def _add_element(self, node: Element):
        self.__neighbors.append(node)

    @property
    def neighbors(self) -> List[Element]:
        return self.__neighbors
