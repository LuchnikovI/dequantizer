from typing import List, Tuple, Dict, Callable, Iterator, Union, Optional, Hashable
from abc import ABC, abstractmethod
from jax import Array
from tensor_node_api import Node
from tensor_edge_api import Edge
from tensor_initializers import tensor_random_normal_initializer
from message_initializer import message_random_nonnegative_initializer

"""An ID of a node can be any hashable object."""
NodeID = Hashable

"""An ID of an edge is the tuple of NodeIDs that are connected by the given edge."""
EdgeID = tuple[NodeID, ...]

"""An ID of a message is the tuple of src and dst of the message."""
MessageID = Union[Tuple[NodeID, EdgeID], Tuple[EdgeID, NodeID]]

"""The tensor network wrapper class."""


class TensorGraph(ABC):
    """Initializes an empty tensor graph."""

    def __init__(self):
        self.__nodes_number: int = 0
        self.__edges_number: int = 0

    """Adds a new node to the tensor graph.
    Args:
        phys_dim: a physical dimension of the node;
        id: an ID of the node, could be any hashable object or None,
            if None the sequential number of the node is used as an ID.
    Returns:
        ID of the node."""

    @abstractmethod
    def add_node(self, phys_dym: int = 2, id: Optional[NodeID] = None) -> NodeID:
        pass

    """Adds a new edge to the tensor graph.
    Args:
        edge_id: an ID of the edge.
    Notes:
        ID of the edge is given by the tuple of nodes IDs that the edge connects.
    """

    @abstractmethod
    def add_edge(self, edge_id: EdgeID, dim: int):
        pass

    """Returns a node given the node ID."""

    @abstractmethod
    def get_node(self, id: NodeID) -> Node:
        pass

    """Returns an edge given the edge ID."""

    @abstractmethod
    def get_edge(self, id: EdgeID) -> Edge:
        pass

    """Returns a depth first search iterator over tensor graph elements (nodes and edges)."""

    @abstractmethod
    def dfs_traversal(self) -> Iterator[Union[Node, Edge]]:
        pass

    """Returns a breadth first search iterator over tensor graph elements (nodes and edges)."""

    @abstractmethod
    def bfs_traversal(self) -> Iterator[Union[Node, Edge]]:
        pass

    """Returns a list with initialized arrays that is used later in pure functions
    performing different algorithms, e.g. belief propagation.
    Args:
        initializer: a closure that takes a node and returns the corresponding tensor.
    Returns:
        A list with initialized tensors.
    Notes:
        Tensors in the list are ordered in accordance with nodes IDs."""

    @abstractmethod
    def init_tensors(self, initializer: Callable[[Node], Array]) -> List[Array]:
        pass

    """Returns a list with initialized messages that is used later in pure functions
    performing different algorithms, e.g. belief propagation.
    Args:
        initializer: a closure that takes an tuple node -> edge or edge -> node and
            returns a corresponding message matrix.
    Returns:
        A dict mapping a message IDs to messages."""

    @abstractmethod
    def init_messages(
        self, initializer: Callable[[Union[Tuple[Node, Edge], Tuple[Edge, Node]]], Array]
    ) -> Dict[MessageID, Array]:
        pass

    """Returns a function that performs one iteration of belief propagation.
    Args:
        traverser: an iterator that traverse elements of a tensor graph (nodes and edges).
    Returns:
        A function that performs one iteration of belief propagation. It takes
            the list of tensors, dict with messages and returns updated messages.
    """

    @abstractmethod
    def get_message_passing_map(
        self, traverser: Iterator[Union[Node, Edge]]
    ) -> Callable[[List[Array], Dict[MessageID, Array]], Dict[MessageID, Array]]:
        pass

    """Returns a function that computes a distance to the vidal gauge.
    The function takes the list of tensors, dict with messages and returns
    distance to the vidal gauge."""

    @abstractmethod
    def get_vidal_gauge_distance_map(
        self,
    ) -> Callable[[List[Array], Dict[MessageID, Array]], Array]:
        pass


"""Returns a random tree tensor graph whose nodes are labeled by
sequential integer numbers starting from 0.
Args:
    nodes_number: nodes number;
    bond_dimension: internal indices dimension.
Return:
    random tree tensor graph."""


def get_random_tree_tensor_graph(
    nodes_number: int,
    bond_dimension: int,
) -> TensorGraph:
    pass


"""Returns an N-dimensional lattice tensor graph whose nodes are labeled
by tuples of the following kind (size0, size1, size2, ...).
Args:
    lattice_sizes: sizes of lattice sides;
    bond_dimension: internal indices dimension.
Returns:
    lattice tensor graph."""


def get_nd_lattice(
    lattice_sizes: List[int],
    bond_dimension: int,
) -> TensorGraph:
    pass


# API testing functions -----------------------------------------------------------------


def small_graph_test(empty_graph: TensorGraph):
    # Hypergraph:
    #
    # Dims  : 2    4    5    3
    # Edges : #    #    #    #
    #         |\  / \  /|   /|\
    #         | \/   \/ |  / | \
    #         | /\   /\ | /  |  \
    #         |/  \ /  \|/   |   \
    # Nodes : @    @    @    @    @
    # Labels: 0    1    2    3    4
    # Dims  : 2    5    3    4    6
    #
    id0 = empty_graph.add_node(2)
    id1 = empty_graph.add_node(5)
    id2 = empty_graph.add_node(3)
    id3 = empty_graph.add_node(4)
    id4 = empty_graph.add_node(6)
    empty_graph.add_edge((0, 1), 2)
    empty_graph.add_edge((2, 0), 4)
    empty_graph.add_edge((1, 2), 5)
    empty_graph.add_edge((4, 2, 3), 6)
    n0 = empty_graph.get_node(0)
    n1 = empty_graph.get_node(1)
    n2 = empty_graph.get_node(2)
    n3 = empty_graph.get_node(3)
    n4 = empty_graph.get_node(4)
    e0 = empty_graph.get_edge((0, 1))
    e1 = empty_graph.get_edge((2, 0))
    e2 = empty_graph.get_edge((1, 2))
    e3 = empty_graph.get_edge((4, 2, 3))
    tensors = empty_graph.init_tensors(tensor_random_normal_initializer)
    messages = empty_graph.init_messages(message_random_nonnegative_initializer)
    # Node IDs correctness
    assert id0 == 0
    assert id1 == 1
    assert id2 == 2
    assert id3 == 3
    assert id4 == 4
    # Physical dimensions correctness
    assert n0.physical_dimension == 2
    assert n1.physical_dimension == 5
    assert n2.physical_dimension == 3
    assert n3.physical_dimension == 4
    assert n4.physical_dimension == 6
    # Tensor ranks correctness
    assert n0.rank == 2
    assert n1.rank == 2
    assert n2.rank == 3
    assert n3.rank == 1
    assert n4.rank == 1
    # Tensor shape correctness
    assert n0.shape == (2, 4)
    assert n1.shape == (2, 5)
    assert n2.shape == (4, 5, 6)
    assert n3.shape == (6,)
    assert n4.shape == (6,)
    # Neighboring edges correctness
    assert n0.indices2edges == [e0, e1]
    assert n1.indices2edges == [e0, e2]
    assert n2.indices2edges == [e1, e2, e3]
    assert n3.indices2edges == [e3]
    assert n4.indices2edges == [e3]
    # Edge dimensions correctness
    assert e0.dimension == 2
    assert e1.dimension == 4
    assert e2.dimension == 5
    assert e3.dimension == 3
    # Neighboring nodes correctness
    assert e0.neighboring_nodes == [n0, n1]
    assert e1.neighboring_nodes == [n2, n0]
    assert e2.neighboring_nodes == [n1, n2]
    assert e3.neighboring_nodes == [n4, n2, n3]
    # Tensors correctness
    assert tensors[0].shape == (2, 4, 2)
    assert tensors[1].shape == (2, 5, 5)
    assert tensors[2].shape == (4, 5, 6, 3)
    assert tensors[3].shape == (6, 4)
    assert tensors[4].shape == (6, 6)
    # Messages correctness
    assert messages[(0, (0, 1))].shape == 2
    assert messages[((0, 1), 0)].shape == 2
    assert messages[(0, (2, 0))].shape == 4
    assert messages[((2, 0), 0)].shape == 4
    assert messages[(1, (0, 1))].shape == 2
    assert messages[((0, 1), 1)].shape == 2
    assert messages[(1, (1, 2))].shape == 5
    assert messages[((1, 2), 1)].shape == 5
    assert messages[(2, (2, 0))].shape == 4
    assert messages[((2, 0), 2)].shape == 4
    assert messages[((1, 2), 2)].shape == 5
    assert messages[(2, (1, 2))].shape == 5
    assert messages[(2, (4, 2, 3))].shape == 6
    assert messages[((4, 2, 3), 2)].shape == 6
    assert messages[(3, (4, 2, 3))].shape == 6
    assert messages[((4, 2, 3), 3)].shape == 6
    assert messages[(4, (4, 2, 3))].shape == 6
    assert messages[((4, 2, 3), 4)].shape == 6
    
