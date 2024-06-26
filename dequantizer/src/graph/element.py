from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Union, Tuple, Hashable, List

"""An ID of a node is any hashable object."""
NodeID = Hashable

"""An ID of an edge is the tuple of NodeIDs that are connected by the given edge."""
EdgeID = tuple[NodeID, ...]

"""An ID of an element is either Node ID or Edge ID."""
ElementID = Union[NodeID, EdgeID]

"""An ID of a message."""


@dataclass(unsafe_hash=True, frozen=True)
class MessageID:
    src: ElementID
    dst: ElementID

    # jax jit requires keys of dict being comparable
    def __lt__(self, other: "MessageID") -> bool:
        return self.__repr__() < other.__repr__()

    def __gt__(self, other: "MessageID") -> bool:
        return self.__repr__() > other.__repr__()

    def __le__(self, other: "MessageID") -> bool:
        return self.__repr__() <= other.__repr__()

    def __ge__(self, other: "MessageID") -> bool:
        return self.__repr__() >= other.__repr__()


"""An element of a tensor graph."""


class Element(ABC):

    def __init__(self, dimension: int, node_id: ElementID):
        self.__id = node_id
        self.__dimension = dimension

    """Gets id of the element."""

    @property
    def id(self) -> ElementID:
        return self.__id

    """Gets dimension of the element."""

    @property
    def dimension(self) -> int:
        return self.__dimension

    @dimension.setter
    def dimension(self, value: int):
        self.__dimension = value

    """Adds a new neighboring element."""

    @abstractmethod
    def _add_element(self, element: "Element"):
        pass

    """Gets neighboring elements list."""

    @property
    @abstractmethod
    def neighbors(self) -> List["Element"]:
        pass

    """Gets degree of the element (number of neighboring elements)."""

    @property
    @abstractmethod
    def degree(self) -> int:
        pass


"""A tuple of two elements is the message direction."""
MessageDir = Tuple[Element, Element]
