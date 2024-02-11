from abc import ABC, abstractmethod
from typing import Union, Tuple, Hashable, List

"""An ID of a node is any hashable object."""
NodeID = Hashable

"""An ID of an edge is the tuple of NodeIDs that are connected by the given edge."""
EdgeID = tuple[NodeID, ...]

"""An ID of an element is either Node ID or Edge ID."""
ElementID = Union[NodeID, EdgeID]

"""An ID of a message is the tuple of src and dst elements."""
MessageID = Tuple[ElementID, ElementID]

"""An element of a tensor graph."""


class Element:

    def __init__(self, dimension: int, id: ElementID):
        self.__id = id
        self.__dimension = dimension

    """Gets id of the element."""

    @property
    def id(self) -> ElementID:
        return self.__id

    """Gets dimension of the element."""

    @property
    def dimension(self) -> int:
        return self.__dimension

    """Adds a new neighboring element."""

    @abstractmethod
    def _add_element(self, element: "Element"):
        pass

    """Gets neighboring elements list."""

    @property
    @abstractmethod
    def neighbors(self) -> List["Element"]:
        pass


"""A tuple of two elements is the message direction."""
MessageDir = Tuple[Element, Element]
