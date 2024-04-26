import sys
from typing import (
    Set,
    List,
    Tuple,
    Dict,
    Callable,
    Iterator,
    Union,
    Optional,
)
from collections import deque
import jax.numpy as jnp
from jax.lax import dynamic_slice
from jax import Array
from jax.random import split, randint, PRNGKey
from .node import Node
from .edge import Edge
from .tensor_initializers import (
    get_tensor_random_normal_initializer,
    get_tensor_std_state_initializer,
)
from .message_initializer import get_message_random_nonnegative_initializer
from .element import NodeID, EdgeID, MessageID, MessageDir
from .tensor_utils import _find_rank
from ..tensor_ops import apply_gate_halves, simple_update, _decompose_gate

"""The tensor network wrapper class."""


class TensorGraph:
    """Initializes an empty tensor graph."""

    def __init__(self):
        self.__default_starting_node: Optional[Node] = None
        self.__nodes: Dict[NodeID, Node] = {}
        self.__edges: Dict[EdgeID, Edge] = {}
        self.__nodes_number: int = 0
        self.__edges_number: int = 0

    """Gets number of nodes in a tensor graph."""

    @property
    def nodes_number(self) -> int:
        return self.__nodes_number

    """Gets number of edges in a tensor graph."""

    @property
    def edges_number(self) -> int:
        return self.__edges_number

    """Adds a new node to the tensor graph.
    Args:
        phys_dim: a physical dimension of the node;
        id: an ID of the node, could be any hashable object or None,
            if None the sequential number of the node is used as an ID.
    Returns:
        ID of the node."""

    def add_node(self, phys_dim: int = 2, node_id: Optional[NodeID] = None) -> NodeID:
        new_id = node_id or self.__nodes_number
        if self.__nodes.get(new_id) is None:
            self.__nodes[new_id] = Node(phys_dim, new_id)
        else:
            raise KeyError(f"Node with ID {new_id} already exists.")
        if self.__nodes_number == 0:
            self.__default_starting_node = self.__nodes[new_id]
        self.__nodes_number += 1
        return new_id

    """Adds a new edge to the tensor graph.
    Args:
        edge_id: an ID of the edge.
    Notes:
        ID of the edge is given by the tuple of nodes IDs that the edge connects.
    """

    def add_edge(self, edge_id: EdgeID, dimension: int):
        if self.__edges.get(edge_id) is None:
            new_edge = Edge(dimension, edge_id)
            for node_id in edge_id:
                connected_node = self.__nodes.get(node_id)
                if connected_node is not None:
                    new_edge._add_element(connected_node)
                    connected_node._add_element(new_edge)
                else:
                    raise KeyError(
                        f"Node with ID {node_id} does not exists in the graph, but presents in the given edge ID {edge_id}"
                    )
            self.__edges[edge_id] = new_edge
        else:
            raise KeyError(f"Edge with ID {edge_id} already exists.")
        self.__edges_number += 1

    """Returns a node given the node ID. If a node with a given ID does not exist, returns None."""

    def get_node(self, node_id: NodeID) -> Optional[Node]:
        return self.__nodes.get(node_id)

    """Returns True if the node with the given ID exists or False if does not."""

    def does_node_exist(self, node_id: NodeID) -> bool:
        return node_id in self.__nodes

    """Returns True if the edge with the given ID exists or False if does not."""

    def does_edge_exist(self, edge_id: EdgeID) -> bool:
        return edge_id in self.__edges

    """Returns an edge given the edge ID. If an edge with a given IF does not exist, returns None."""

    def get_edge(self, edge_id: EdgeID) -> Optional[Edge]:
        return self.__edges.get(edge_id)

    # """Returns Edge IDs corresponding to the spanning tree that covers
    # maximal sum of bond dimensions."""

    # def get_maximal_spanning_tree(self) -> Set[EdgeID]:
    #    spanning_tree: Set[EdgeID] = set()
    #    connected_nodes: List[Node] = [ self.__nodes.items().__iter__().__next__()[1] ]
    #    cut_edges: Set[Edge] = {  }
    #    raise NotImplementedError()

    """Returns an iterator over tensor graph elements (nodes and edges).
    Args:
        starting_id: either node or edge from which the traversal starts.
            If None, first added node to the graph is used as the starting node;
        ordering: string specifying order of traversal. Currently the following
            traversal methods are supported:
                "bfs": stands for breadth first search;
                "dfs": stands for depth first search.
    Returns:
        Iterator iterating edges and nodes. If the graph is empty returns None."""

    def get_traversal_iterator(
        self,
        starting_element: Optional[Union[Node, Edge]] = None,
        ordering: str = "bfs",
    ) -> Optional[Iterator[Union[Node, Edge]]]:
        stack: deque = deque()
        pop_fn: Callable[[deque[Union[Node, Edge]]], Union[Node, Edge]]
        if ordering == "bfs":

            def pop_fn(d):
                return d.popleft()

        elif ordering == "dfs":

            def pop_fn(d):
                return d.pop()

        else:
            # TODO: custom exception for this error
            raise KeyError(f"Nodes / edges ordering method {ordering} is unknown.")
        visited_ids: Set[Union[NodeID, EdgeID]] = set()
        if starting_element is None:
            if self.__default_starting_node is None:
                return None
            stack.append(self.__default_starting_node)
        else:
            stack.append(starting_element)
        while len(stack) != 0:
            element = pop_fn(stack)
            if element.id not in visited_ids:
                yield element
                visited_ids.add(element.id)
                for new_element in element.neighbors:
                    stack.append(new_element)

    """Returns a dict with initialized node tensors that is used later in pure functions
    performing different algorithms, e.g. belief propagation.
    Args:
        initializer: a closure that takes a node and returns a corresponding
            node tensor.
    Returns:
        A dict mapping a node IDs to tensors."""

    def init_tensors(self, initializer: Callable[[Node], Array]) -> Dict[NodeID, Array]:
        tensors_dict: Dict[NodeID, Array] = {}
        elements_iterator = self.get_traversal_iterator()
        if elements_iterator is None:
            return tensors_dict
        for element in elements_iterator:
            if isinstance(element, Node):
                tensor = initializer(element)
                expected_shape = (*element.bond_shape, element.dimension)
                if expected_shape == tensor.shape:
                    tensors_dict[element.id] = tensor
                else:
                    raise ValueError(
                        f"Expected tensor of shape {expected_shape} got tensor of shape {tensor.shape}."
                    )
        return tensors_dict

    """Returns a dict with initialized messages that is used later in pure functions
    performing different algorithms, e.g. belief propagation.
    Args:
        initializer: a closure that takes a tuple node -> edge or edge -> node and
            returns a corresponding message matrix.
    Returns:
        A dict mapping a message IDs to messages."""

    def init_messages(
        self,
        initializer: Callable[[MessageDir], Array],
    ) -> Dict[MessageID, Array]:
        message_dict: Dict[MessageID, Array] = {}
        elements_iterator = self.get_traversal_iterator()
        if elements_iterator is None:
            return message_dict
        else:
            for element in elements_iterator:
                if isinstance(element, Node):
                    for edge in element.neighbors:
                        message = initializer((element, edge))
                        message_dict[MessageID(src=element.id, dst=edge.id)] = message
                elif isinstance(element, Edge):
                    for node in element.neighbors:
                        message = initializer((element, node))
                        message_dict[MessageID(src=element.id, dst=node.id)] = message
        return message_dict

    """Truncates the given graph inplace and corresponding tensors and core edge tensors.
    Args:
        tensors: node tensors;
        core_edge_tensors: core edge tensors;
        accuracy: the maximal truncation error, not considered if None;
        max_rank: the maximal allowed edge dimension, not considered if None.
    Returns:
        tuple with truncated node tensors and truncated core edge tensors.
    """

    def truncate(
        self,
        tensors: Dict[NodeID, Array],
        core_edge_tensors: Dict[EdgeID, Array],
        accuracy: Optional[Union[float, Array]],
        max_rank: Optional[int] = None,
    ) -> Tuple[Dict[NodeID, Array], Dict[EdgeID, Array]]:
        truncated_core_edge_tensors: Dict[EdgeID, Array] = {}
        truncated_tensors: Dict[NodeID, Array] = {}
        for edge_id, core_edge_tensor in core_edge_tensors.items():
            rank = _find_rank(core_edge_tensor, accuracy, max_rank)
            truncated_core_edge_tensors[edge_id] = core_edge_tensor[:rank]
            self.__edges[edge_id].dimension = int(rank)
        for node_id, node in self.__nodes.items():
            tensor = tensors[node_id]
            new_bond_shape: List[int] = []
            for neighbor in node.neighbors:
                new_bond_shape.append(neighbor.dimension)
            node.bond_shape = tuple(new_bond_shape)
            truncated_tensor = dynamic_slice(
                tensor, (node.degree + 1) * [0], new_bond_shape + [tensor.shape[-1]]
            )
            truncated_tensors[node_id] = truncated_tensor
        return truncated_tensors, truncated_core_edge_tensors

    """Applies a single-side gate to a given node.
    Args:
        node_id: id of a node where to apply a gate;
        tensors: node tensors;
        gate: a matrix specifying a single-side gate.
    Returns:
        updated tensors."""

    def apply1(
        self, node_id: NodeID, tensors: Dict[NodeID, Array], gate: Array
    ) -> Dict[NodeID, Array]:
        tensor = tensors[node_id]
        tensor = jnp.tensordot(tensor, gate, axes=[-1, 1])
        tensor /= jnp.linalg.norm(tensor)
        tensors[node_id] = tensor
        return tensors

    """Applies a two-sides gate to a Vidal canonical form, truncates the result if necessary.
    It preserve the Vidal canonical form up to the truncation error.
    Args:
        controlling_node_id: first node affected by a gate (can be seen as a controlling node);
        controlled_node_id: second node affected by a gate (can be seen as a controlled node);
        tensors: node tensors;
        lambdas: singular values that are placed on the edges;
        gate: a gate that is being applied to the canonical form;
        threshold: a small value used for regularization purposes;
        accuracy: the maximal truncation error, not considered if None;
        max_rank: the maximal allowed edge dimension, not considered if None.
    Returns:
        updated tensors, singular values and resulting truncation error (
            normalized sqrt of sum of squares of singular values),
            entropy of lambdas sitting on the edge.
    """

    def apply2_to_vidal_canonical(
        self,
        controlling_node_id: NodeID,
        controlled_node_id: NodeID,
        tensors: Dict[NodeID, Array],
        lambdas: Dict[EdgeID, Array],
        gate: Array,
        threshold: Union[Array, float],
        accuracy: Optional[Union[float, Array]],
        max_rank: Optional[int] = None,
    ) -> Tuple[Dict[NodeID, Array], Dict[EdgeID, Array], Array, Array]:
        if len(gate.shape) != 4:
            raise ValueError("Gate must be a tensor of rank 4.")
        if gate.shape[3] != gate.shape[1] or gate.shape[2] != gate.shape[0]:
            raise ValueError("Input - output dimensions must match with each other.")
        trial_id1 = (controlled_node_id, controlling_node_id)
        trial_id2 = (controlling_node_id, controlled_node_id)
        edge = self.get_edge(trial_id1) or self.get_edge(trial_id2)
        if edge is None:
            raise ValueError(
                f"Neither edge with ID {trial_id1} nor with ID {trial_id2} is found in the tensor graph."
            )
        if edge.degree != 2:
            raise NotImplementedError("apply2 is implemented only for two qubits.")
        assert len(edge.neighbors) == edge.degree
        node1 = self.get_node(controlling_node_id)
        node2 = self.get_node(controlled_node_id)
        if not isinstance(node1, Node) or not isinstance(node2, Node):
            raise NotImplementedError(
                "This branch is unreachable if the code is correct."
            )
        tensor1 = tensors[node1.id]
        tensor2 = tensors[node2.id]
        edge_id = edge.id
        if not isinstance(edge_id, tuple):
            raise NotImplementedError(
                "This branch is unreachable if the code is correct."
            )
        index1 = node1.indices[edge_id]
        index2 = node2.indices[edge_id]
        half1, half2 = _decompose_gate(gate, threshold)
        lambdas1 = []
        for neighbor in node1.neighbors:
            neighbor_id = neighbor.id
            if isinstance(neighbor_id, tuple):
                lambdas1.append(lambdas[neighbor_id])
            else:
                raise NotImplementedError(
                    "This branch is unreachable if the code is correct."
                )
        lambdas2 = []
        for neighbor in node2.neighbors:
            neighbor_id = neighbor.id
            if isinstance(neighbor_id, tuple):
                lambdas2.append(lambdas[neighbor_id])
            else:
                raise NotImplementedError(
                    "This branch is unreachable if the code is correct."
                )
        tensor1, tensor2, lmbd, err, entropy = simple_update(
            tensor1,
            tensor2,
            lambdas1,
            lambdas2,
            half1,
            half2,
            index1,
            index2,
            threshold,
            accuracy,
            max_rank,
        )
        tensors[node1.id] = tensor1
        tensors[node2.id] = tensor2
        edge.dimension = lmbd.shape[0]
        assert lmbd.shape[0] == tensor1.shape[index1]
        assert lmbd.shape[0] == tensor2.shape[index2]
        assert lmbd.shape[0] == edge.dimension
        node1.bond_shape = tensor1.shape[:-1]
        node2.bond_shape = tensor2.shape[:-1]
        lambdas[edge_id] = lmbd
        return tensors, lambdas, err, entropy

    """Applies a two-sides gate to a given subset of sides connected but the given edge.
    Args:
        controlling_node_id: first node affected by a gate (can be seen as a controlling node);
        controlled_node_id: second node affected by a gate (can be seen as a controlled node);
        tensors: node tensors;
        gate: rank 4 tensor that represents a gate;
        threshold: threshold of gate truncation.
    Returns:
        updated tensors, note also, that this method modifies a network itself."""

    def apply2(
        self,
        controlling_node_id: NodeID,
        controlled_node_id: NodeID,
        tensors: Dict[NodeID, Array],
        gate: Array,
        threshold: Union[Array, float],
    ) -> Dict[NodeID, Array]:
        if len(gate.shape) != 4:
            raise ValueError("Gate must be a tensor of rank 4.")
        if gate.shape[3] != gate.shape[1] or gate.shape[2] != gate.shape[0]:
            raise ValueError("Input - output dimensions must match with each other.")
        trial_id1 = (controlled_node_id, controlling_node_id)
        trial_id2 = (controlling_node_id, controlled_node_id)
        edge = self.get_edge(trial_id1) or self.get_edge(trial_id2)
        if edge is None:
            raise ValueError(
                f"Neither edge with ID {trial_id1} nor with ID {trial_id2} is found in the tensor graph."
            )
        if edge.degree != 2:
            raise NotImplementedError("apply2 is implemented only for two qubits.")
        assert len(edge.neighbors) == edge.degree
        node1 = self.get_node(controlling_node_id)
        node2 = self.get_node(controlled_node_id)
        if not isinstance(node1, Node) or not isinstance(node2, Node):
            raise NotImplementedError(
                "This branch is unreachable if the code is correct."
            )
        tensor1 = tensors[node1.id]
        tensor2 = tensors[node2.id]
        edge_id = edge.id
        if not isinstance(edge_id, tuple):
            raise NotImplementedError(
                "This branch is unreachable if the code is correct."
            )
        index1 = node1.indices[edge_id]
        index2 = node2.indices[edge_id]
        half1, half2 = _decompose_gate(gate, threshold)
        gate_rank = half1.shape[0]
        tensor1, tensor2 = apply_gate_halves(
            tensor1, tensor2, index1, index2, half1, half2
        )
        tensors[node1.id] = tensor1
        tensors[node2.id] = tensor2
        edge.dimension = gate_rank * edge.dimension
        node1.bond_shape = tensor1.shape[:-1]
        node2.bond_shape = tensor2.shape[:-1]
        assert (
            edge.dimension == tensors[node1.id].shape[index1]
        ), f"{edge.dimension}, {tensors[node1.id].shape[index1]}"
        assert (
            edge.dimension == tensors[node2.id].shape[index2]
        ), f"{edge.dimension}, {tensors[node2.id].shape[index2]}"
        assert (
            node1.bond_shape == tensors[node1.id].shape[:-1]
        ), f"{node1.bond_shape}, {tensors[node1.id].shape[:-1]}"
        assert (
            node2.bond_shape == tensors[node2.id].shape[:-1]
        ), f"{node2.bond_shape}, {tensors[node2.id].shape[:-1]}"
        return tensors

    """Sets all bond dimensions to 1."""

    def to_product(self):
        for node in self.__nodes.values():
            node.bond_shape = len(node.bond_shape) * (1,)
            assert self.get_node(node.id).bond_shape == len(node.bond_shape) * (1,)
        for edge in self.__edges.values():
            edge.dimension = 1
            assert self.get_edge(edge.id).dimension == 1


"""Returns a random tree tensor graph whose nodes are labeled by
sequential integer numbers starting from 0.
Args:
    nodes_number: nodes number;
    phys_dimension: dimension of the physical space per node;
    bond_dimensions: list of possible bond dimensions sampled randomly;
    key: jax random number generator seed.
Return:
    random tree tensor graph."""


def get_random_tree_tensor_graph(
    nodes_number: int,
    phys_dimension: int,
    bond_dimensions: List[int],
    key: Array,
) -> TensorGraph:
    random_tree = TensorGraph()
    # TODO: move from list to hash tables (set ?) to improve asymptotic
    connected_nodes = [0]
    disconnected_nodes = list(range(1, nodes_number))
    random_tree.add_node(phys_dimension, 0)
    for _ in range(1, nodes_number):
        disconnected_set_size = len(disconnected_nodes)
        connected_set_size = len(connected_nodes)
        key, subkey = split(key)
        connected_id = connected_nodes[
            int(randint(subkey, (1,), 0, connected_set_size)[0])
        ]
        key, subkey = split(key)
        disconnected_id = disconnected_nodes[
            int(randint(subkey, (1,), 0, disconnected_set_size)[0])
        ]
        random_tree.add_node(phys_dimension, disconnected_id)
        key, subkey = split(key)
        idx = randint(subkey, shape=(1,), minval=0, maxval=len(bond_dimensions))[0]
        random_tree.add_edge((connected_id, disconnected_id), bond_dimensions[idx])
        disconnected_nodes.remove(disconnected_id)
        connected_nodes.append(disconnected_id)
    return random_tree


"""Returns an N-dimensional lattice tensor graph whose nodes are labeled
by tuples of the following kind (size0, size1, size2, ...).
Args:
    lattice_sizes: sizes of lattice sides;
    bond_dimension: internal indices dimension;
    open_boundary: indicates if to use open boundary conditions or periodic.
Returns:
    lattice tensor graph."""


def get_nd_lattice(
    lattice_sizes: List[int],
    phys_dimension: int,
    bond_dimension: int,
    open_boundary: bool = True,
) -> TensorGraph:
    nodes_number = 1
    for size in lattice_sizes:
        nodes_number *= size
    lattice = TensorGraph()
    node_id: List[int]
    ids: List[List[int]] = []
    for i in range(nodes_number):
        node_id = []
        for size in lattice_sizes:
            node_id.append(i % size)
            i //= size
        lattice.add_node(phys_dimension, tuple(node_id))
        ids.append(node_id)
    for j in ids:
        for pos in range(len(j)):
            next_j = j.copy()
            next_j[pos] += 1
            if lattice.does_node_exist(tuple(next_j)):
                lattice.add_edge((tuple(j), tuple(next_j)), bond_dimension)
            elif not open_boundary:
                next_j[pos] = 0
                lattice.add_edge((tuple(j), tuple(next_j)), bond_dimension)
    return lattice


"""Returns a tensor graph corresponding to the heavy hex lattice architecture
of IBM Eagle processor with 127 qubits.
See Fig. 1 from https://arxiv.org/abs/2306.14887 for a reference."""


def get_heavy_hex_ibm_eagle_lattice() -> TensorGraph:
    lattice = TensorGraph()
    for _ in range(127):
        lattice.add_node()
    for i in range(13):
        lattice.add_edge((i, i + 1), 1)
    for i in range(18, 32):
        lattice.add_edge((i, i + 1), 1)
    for i in range(37, 51):
        lattice.add_edge((i, i + 1), 1)
    for i in range(56, 70):
        lattice.add_edge((i, i + 1), 1)
    for i in range(75, 89):
        lattice.add_edge((i, i + 1), 1)
    for i in range(94, 108):
        lattice.add_edge((i, i + 1), 1)
    for i in range(113, 126):
        lattice.add_edge((i, i + 1), 1)
    lattice.add_edge((0, 14), 1)
    lattice.add_edge((14, 18), 1)
    lattice.add_edge((4, 15), 1)
    lattice.add_edge((15, 22), 1)
    lattice.add_edge((8, 16), 1)
    lattice.add_edge((16, 26), 1)
    lattice.add_edge((12, 17), 1)
    lattice.add_edge((17, 30), 1)
    lattice.add_edge((20, 33), 1)
    lattice.add_edge((33, 39), 1)
    lattice.add_edge((24, 34), 1)
    lattice.add_edge((34, 43), 1)
    lattice.add_edge((28, 35), 1)
    lattice.add_edge((35, 47), 1)
    lattice.add_edge((32, 36), 1)
    lattice.add_edge((36, 51), 1)
    lattice.add_edge((37, 52), 1)
    lattice.add_edge((52, 56), 1)
    lattice.add_edge((41, 53), 1)
    lattice.add_edge((53, 60), 1)
    lattice.add_edge((45, 54), 1)
    lattice.add_edge((54, 64), 1)
    lattice.add_edge((49, 55), 1)
    lattice.add_edge((55, 68), 1)
    lattice.add_edge((58, 71), 1)
    lattice.add_edge((71, 77), 1)
    lattice.add_edge((62, 72), 1)
    lattice.add_edge((72, 81), 1)
    lattice.add_edge((66, 73), 1)
    lattice.add_edge((73, 85), 1)
    lattice.add_edge((70, 74), 1)
    lattice.add_edge((74, 89), 1)
    lattice.add_edge((75, 90), 1)
    lattice.add_edge((90, 94), 1)
    lattice.add_edge((79, 91), 1)
    lattice.add_edge((91, 98), 1)
    lattice.add_edge((83, 92), 1)
    lattice.add_edge((92, 102), 1)
    lattice.add_edge((87, 93), 1)
    lattice.add_edge((93, 106), 1)
    lattice.add_edge((96, 109), 1)
    lattice.add_edge((109, 114), 1)
    lattice.add_edge((100, 110), 1)
    lattice.add_edge((110, 118), 1)
    lattice.add_edge((104, 111), 1)
    lattice.add_edge((111, 122), 1)
    lattice.add_edge((108, 112), 1)
    lattice.add_edge((112, 126), 1)
    return lattice


"""Returns a tensor graph corresponding to the heavy hex lattice architecture
of IBM Eagle processor with infinite number of qubits.
See Fig. 5 from https://arxiv.org/abs/2306.14887 for a reference."""


def get_heavy_hex_ibm_eagle_lattice_infinite() -> TensorGraph:
    lattice = TensorGraph()
    for _ in range(5):
        lattice.add_node(2)
    lattice.add_edge((0, 1), 1)
    lattice.add_edge((1, 2), 1)
    lattice.add_edge((2, 3), 1)
    lattice.add_edge((2, 4), 1)
    lattice.add_edge((3, 0), 1)
    lattice.add_edge((4, 0), 1)
    return lattice


# API testing functions -----------------------------------------------------------------


def small_graph_test(empty_graph: TensorGraph, key: Array):
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
    assert n0 is not None
    assert n1 is not None
    assert n2 is not None
    assert n3 is not None
    assert n4 is not None
    e0 = empty_graph.get_edge((0, 1))
    e1 = empty_graph.get_edge((2, 0))
    e2 = empty_graph.get_edge((1, 2))
    e3 = empty_graph.get_edge((4, 2, 3))
    assert e0 is not None
    assert e1 is not None
    assert e2 is not None
    assert e3 is not None
    tensor_initializer = get_tensor_random_normal_initializer(key)
    _, key = split(key)
    message_initializer = get_message_random_nonnegative_initializer(key)
    tensors = empty_graph.init_tensors(tensor_initializer)
    messages = empty_graph.init_messages(message_initializer)
    # Number of nodes / edges correctness
    assert empty_graph.nodes_number == 5
    assert empty_graph.edges_number == 4
    print("Nodes / edges number: OK", file=sys.stderr)
    # Node IDs correctness
    assert id0 == 0
    assert id1 == 1
    assert id2 == 2
    assert id3 == 3
    assert id4 == 4
    print("Node IDs: OK", file=sys.stderr)
    # Physical dimensions correctness
    assert n0.dimension == 2
    assert n1.dimension == 5
    assert n2.dimension == 3
    assert n3.dimension == 4
    assert n4.dimension == 6
    print("Node dimensions: OK", file=sys.stderr)
    # Node / edge degree correctness
    assert n0.degree == 2
    assert n1.degree == 2
    assert n2.degree == 3
    assert n3.degree == 1
    assert n4.degree == 1
    assert e0.degree == 2
    assert e1.degree == 2
    assert e2.degree == 2
    assert e3.degree == 3
    print("Node / edge degrees: OK", file=sys.stderr)
    # Tensor shape correctness
    assert n0.bond_shape == (2, 4)
    assert n1.bond_shape == (2, 5)
    assert n2.bond_shape == (4, 5, 6)
    assert n3.bond_shape == (6,)
    assert n4.bond_shape == (6,)
    print("Bond shapes: OK", file=sys.stderr)
    # Neighboring edges correctness
    assert n0.neighbors == [e0, e1]
    assert n1.neighbors == [e0, e2]
    assert n2.neighbors == [e1, e2, e3]
    assert n3.neighbors == [e3]
    assert n4.neighbors == [e3]
    print("Neighboring edges: OK", file=sys.stderr)
    # Edge dimensions correctness
    assert e0.dimension == 2
    assert e1.dimension == 4
    assert e2.dimension == 5
    assert e3.dimension == 6
    print("Edge dimensions: OK", file=sys.stderr)
    # Neighboring nodes correctness
    assert e0.neighbors == [n0, n1]
    assert e1.neighbors == [n2, n0]
    assert e2.neighbors == [n1, n2]
    assert e3.neighbors == [n4, n2, n3]
    print("Neighboring nodes: OK", file=sys.stderr)
    # Tensors correctness
    assert tensors[0].shape == (2, 4, 2)
    assert tensors[1].shape == (2, 5, 5)
    assert tensors[2].shape == (4, 5, 6, 3)
    assert tensors[3].shape == (6, 4)
    assert tensors[4].shape == (6, 6)
    assert len(tensors) == 5
    print("Initialized tensor shapes: OK", file=sys.stderr)
    # Messages correctness
    assert messages[MessageID(0, (0, 1))].shape == (2, 2)
    assert messages[MessageID((0, 1), 0)].shape == (2, 2)
    assert messages[MessageID(0, (2, 0))].shape == (4, 4)
    assert messages[MessageID((2, 0), 0)].shape == (4, 4)
    assert messages[MessageID(1, (0, 1))].shape == (2, 2)
    assert messages[MessageID((0, 1), 1)].shape == (2, 2)
    assert messages[MessageID(1, (1, 2))].shape == (5, 5)
    assert messages[MessageID((1, 2), 1)].shape == (5, 5)
    assert messages[MessageID(2, (2, 0))].shape == (4, 4)
    assert messages[MessageID((2, 0), 2)].shape == (4, 4)
    assert messages[MessageID((1, 2), 2)].shape == (5, 5)
    assert messages[MessageID(2, (1, 2))].shape == (5, 5)
    assert messages[MessageID(2, (4, 2, 3))].shape == (6, 6)
    assert messages[MessageID((4, 2, 3), 2)].shape == (6, 6)
    assert messages[MessageID(3, (4, 2, 3))].shape == (6, 6)
    assert messages[MessageID((4, 2, 3), 3)].shape == (6, 6)
    assert messages[MessageID(4, (4, 2, 3))].shape == (6, 6)
    assert messages[MessageID((4, 2, 3), 4)].shape == (6, 6)
    m = messages[MessageID((4, 2, 3), 4)]
    assert (jnp.linalg.eigvalsh(m) > -1e-5).all()
    assert len(messages) == 18
    print("Initialized message shapes: OK", file=sys.stderr)
    #  Elements traversal
    empty_graph.add_node()
    nodes_subset = {
        n0,
        n1,
        n2,
        n3,
        n4,
    }
    edges_subset = {
        e0,
        e1,
        e2,
        e3,
    }
    elements_set = set()
    for element in empty_graph.get_traversal_iterator(e1, "dfs") or iter([]):
        elements_set.add(element)
    assert elements_set.issuperset(nodes_subset)
    assert elements_set.issuperset(edges_subset)
    assert len(elements_set) == 9
    elements_set = set()
    for element in empty_graph.get_traversal_iterator(n2, "bfs") or iter([]):
        elements_set.add(element)
    assert elements_set.issuperset(nodes_subset)
    assert elements_set.issuperset(edges_subset)
    assert len(elements_set) == 9
    print("Elements traversal: OK", file=sys.stderr)


def lattice_3d_test(
    sizes: Tuple[int, int, int], bond_dim: int, phys_dim: int, key: Array
):
    lattice = get_nd_lattice(list(sizes), phys_dim, bond_dim)
    tensor_initializer = get_tensor_random_normal_initializer(key)
    _, key = split(key)
    messages_initializer = get_message_random_nonnegative_initializer(key)
    tensors = lattice.init_tensors(tensor_initializer)
    messages = lattice.init_messages(messages_initializer)
    # Checking corner tensors of the lattice
    assert tensors[(0, 0, 0)].shape == (bond_dim, bond_dim, bond_dim, phys_dim)
    assert tensors[(sizes[0] - 1, 0, 0)].shape == (
        bond_dim,
        bond_dim,
        bond_dim,
        phys_dim,
    )
    assert tensors[(0, sizes[1] - 1, 0)].shape == (
        bond_dim,
        bond_dim,
        bond_dim,
        phys_dim,
    )
    assert tensors[(sizes[0] - 1, sizes[1] - 1, 0)].shape == (
        bond_dim,
        bond_dim,
        bond_dim,
        phys_dim,
    )
    assert tensors[(0, 0, sizes[2] - 1)].shape == (
        bond_dim,
        bond_dim,
        bond_dim,
        phys_dim,
    )
    assert tensors[(sizes[0] - 1, 0, sizes[2] - 1)].shape == (
        bond_dim,
        bond_dim,
        bond_dim,
        phys_dim,
    )
    assert tensors[(0, sizes[1] - 1, sizes[2] - 1)].shape == (
        bond_dim,
        bond_dim,
        bond_dim,
        phys_dim,
    )
    assert tensors[(sizes[0] - 1, sizes[1] - 1, sizes[2] - 1)].shape == (
        bond_dim,
        bond_dim,
        bond_dim,
        phys_dim,
    )
    print("Corner tensor shapes: OK", file=sys.stderr)
    # Checking 1d edges of the lattice
    for i in range(1, sizes[0] - 2):
        assert tensors[(i, 0, 0)].shape == (
            bond_dim,
            bond_dim,
            bond_dim,
            bond_dim,
            phys_dim,
        )
        assert tensors[(i, sizes[1] - 1, 0)].shape == (
            bond_dim,
            bond_dim,
            bond_dim,
            bond_dim,
            phys_dim,
        )
        assert tensors[(i, 0, sizes[2] - 1)].shape == (
            bond_dim,
            bond_dim,
            bond_dim,
            bond_dim,
            phys_dim,
        )
        assert tensors[(i, sizes[1] - 1, sizes[2] - 1)].shape == (
            bond_dim,
            bond_dim,
            bond_dim,
            bond_dim,
            phys_dim,
        )
    for i in range(1, sizes[1] - 2):
        assert tensors[(0, i, 0)].shape == (
            bond_dim,
            bond_dim,
            bond_dim,
            bond_dim,
            phys_dim,
        )
        assert tensors[(sizes[0] - 1, i, 0)].shape == (
            bond_dim,
            bond_dim,
            bond_dim,
            bond_dim,
            phys_dim,
        )
        assert tensors[(0, i, sizes[2] - 1)].shape == (
            bond_dim,
            bond_dim,
            bond_dim,
            bond_dim,
            phys_dim,
        )
        assert tensors[(sizes[0] - 1, i, sizes[2] - 1)].shape == (
            bond_dim,
            bond_dim,
            bond_dim,
            bond_dim,
            phys_dim,
        )
    for i in range(1, sizes[2] - 2):
        assert tensors[(0, 0, i)].shape == (
            bond_dim,
            bond_dim,
            bond_dim,
            bond_dim,
            phys_dim,
        )
        assert tensors[(sizes[0] - 1, 0, i)].shape == (
            bond_dim,
            bond_dim,
            bond_dim,
            bond_dim,
            phys_dim,
        )
        assert tensors[(0, sizes[1] - 1, i)].shape == (
            bond_dim,
            bond_dim,
            bond_dim,
            bond_dim,
            phys_dim,
        )
        assert tensors[(sizes[0] - 1, sizes[1] - 1, i)].shape == (
            bond_dim,
            bond_dim,
            bond_dim,
            bond_dim,
            phys_dim,
        )
    print("1d edge tensor shapes: OK", file=sys.stderr)
    # Checking 2d edges of the lattice
    for i in range(1, sizes[0] - 2):
        for j in range(1, sizes[1] - 2):
            assert tensors[(i, j, 0)].shape == (
                bond_dim,
                bond_dim,
                bond_dim,
                bond_dim,
                bond_dim,
                phys_dim,
            )
            assert tensors[(i, j, sizes[2] - 1)].shape == (
                bond_dim,
                bond_dim,
                bond_dim,
                bond_dim,
                bond_dim,
                phys_dim,
            )
    for i in range(1, sizes[0] - 2):
        for j in range(1, sizes[2] - 2):
            assert tensors[(i, 0, j)].shape == (
                bond_dim,
                bond_dim,
                bond_dim,
                bond_dim,
                bond_dim,
                phys_dim,
            )
            assert tensors[(i, sizes[1] - 1, j)].shape == (
                bond_dim,
                bond_dim,
                bond_dim,
                bond_dim,
                bond_dim,
                phys_dim,
            )
    for i in range(1, sizes[1] - 2):
        for j in range(1, sizes[2] - 2):
            assert tensors[(0, i, j)].shape == (
                bond_dim,
                bond_dim,
                bond_dim,
                bond_dim,
                bond_dim,
                phys_dim,
            )
            assert tensors[(sizes[0] - 1, i, j)].shape == (
                bond_dim,
                bond_dim,
                bond_dim,
                bond_dim,
                bond_dim,
                phys_dim,
            )
    print("2d edge tensor shapes: OK", file=sys.stderr)
    # Checking lattice internal tensors
    for i in range(1, sizes[0] - 2):
        for j in range(1, sizes[1] - 2):
            for k in range(1, sizes[2] - 2):
                assert tensors[(i, j, k)].shape == (
                    bond_dim,
                    bond_dim,
                    bond_dim,
                    bond_dim,
                    bond_dim,
                    bond_dim,
                    phys_dim,
                )
    print("Internal tensor shapes: OK", file=sys.stderr)
    # Checking nodes / edges / messages number
    nodes_number = sizes[0] * sizes[1] * sizes[2]
    edges_number = (
        (sizes[0] - 1) * sizes[1] * sizes[2]
        + sizes[0] * (sizes[1] - 1) * sizes[2]
        + sizes[0] * sizes[1] * (sizes[2] - 1)
    )
    messages_number = 4 * edges_number
    edges_set = set()
    nodes_set = set()
    for element in lattice.get_traversal_iterator() or iter([]):
        if isinstance(element, Node):
            nodes_set.add(element)
        elif isinstance(element, Edge):
            edges_set.add(element)
        else:
            raise NotImplementedError(
                "This is unreachable branch if the code is correct."
            )
    assert len(nodes_set) == nodes_number
    assert len(tensors) == nodes_number
    assert len(messages) == messages_number
    assert len(edges_set) == edges_number
    print("Nodes / Edges / Messages number: OK")
    # Checking phys. dimension and bond dimension
    for element in lattice.get_traversal_iterator() or iter([]):
        if isinstance(element, Node):
            element.dimension == phys_dim
        elif isinstance(element, Edge):
            element.dimension == bond_dim
    print("Physical dimensions and bond dimensions: OK")


def random_tree_test(
    nodes_number: int,
    phys_dimension: int,
    bond_dimensions: List[int],
    key: Array,
):
    tree = get_random_tree_tensor_graph(
        nodes_number, phys_dimension, bond_dimensions, key
    )
    # Checking nodes / edges number
    assert tree.nodes_number == nodes_number
    assert tree.edges_number == nodes_number - 1
    print("Nodes / Edges number: OK")
    # Checking connectivity / phys. dimension / bond dimension
    traversed_nodes_number = 0
    traversed_edges_number = 0
    for element in tree.get_traversal_iterator() or iter([]):
        if isinstance(element, Node):
            assert element.dimension == phys_dimension
            traversed_nodes_number += 1
        elif isinstance(element, Edge):
            traversed_edges_number += 1
        else:
            raise NotImplementedError(
                "This is unreachable branch if the code is correct."
            )
    assert traversed_nodes_number == nodes_number
    assert traversed_edges_number == nodes_number - 1
    print("Connectivity / phys. dimension / bond dimension: OK")


def ghz_state_preparation_test():
    from ..mappings.belief_propagation import get_belief_propagation_map
    from ..mappings.vidal_gauge import get_vidal_gauge_fixing_map
    from ..mappings.symmetric_gauge import get_symmetric_gauge_fixing_map
    from ..mappings.messages_distance import messages_frob_distance
    from ..mappings.density_matrix import get_one_side_density_matrix

    lattice = get_nd_lattice([4, 4], 2, 1)
    print(lattice.nodes_number)
    tensor_initializer = get_tensor_std_state_initializer()
    tensors = lattice.init_tensors(tensor_initializer)
    hadamard = jnp.array(
        [
            [1, 1],
            [1, -1],
        ],
        dtype=jnp.complex128,
    )
    cnot = jnp.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=jnp.complex128,
    ).reshape((2, 2, 2, 2))
    tensors = lattice.apply1((0, 0), tensors, hadamard)
    tensors = lattice.apply2((0, 0), (0, 1), tensors, cnot, 1e-8)
    tensors = lattice.apply2((0, 1), (0, 2), tensors, cnot, 1e-8)
    tensors = lattice.apply2((0, 2), (0, 3), tensors, cnot, 1e-8)
    tensors = lattice.apply2((0, 0), (1, 0), tensors, cnot, 1e-8)
    tensors = lattice.apply2((0, 1), (1, 1), tensors, cnot, 1e-8)
    tensors = lattice.apply2((0, 2), (1, 2), tensors, cnot, 1e-8)
    tensors = lattice.apply2((0, 3), (1, 3), tensors, cnot, 1e-8)
    tensors = lattice.apply2((1, 0), (2, 0), tensors, cnot, 1e-8)
    tensors = lattice.apply2((2, 0), (2, 1), tensors, cnot, 1e-8)
    tensors = lattice.apply2((2, 1), (2, 2), tensors, cnot, 1e-8)
    tensors = lattice.apply2((2, 2), (2, 3), tensors, cnot, 1e-8)
    tensors = lattice.apply2((2, 3), (3, 3), tensors, cnot, 1e-8)
    tensors = lattice.apply2((3, 3), (3, 2), tensors, cnot, 1e-8)
    tensors = lattice.apply2((2, 1), (3, 1), tensors, cnot, 1e-8)
    tensors = lattice.apply2((2, 0), (3, 0), tensors, cnot, 1e-8)
    message_initializer = get_message_random_nonnegative_initializer(PRNGKey(42))
    messages = lattice.init_messages(message_initializer)
    dist = jnp.finfo(jnp.float64).max
    traverser = list(lattice.get_traversal_iterator() or iter([]))
    bp_map = get_belief_propagation_map(traverser)
    vg_map = get_vidal_gauge_fixing_map(traverser, 1e-8)
    sg_map = get_symmetric_gauge_fixing_map(traverser, 1e-8)
    while dist > 1e-8:
        new_messages = bp_map(tensors, messages)
        dist = messages_frob_distance(new_messages, messages)
        messages = new_messages
    for i in range(4):
        for j in range(4):
            dens = get_one_side_density_matrix(
                lattice.get_node((i, j)), tensors, messages
            )
            assert jnp.isclose(dens, jnp.eye(2) / 2).all()
    tensors = lattice.apply2((2, 0), (3, 0), tensors, cnot, 1e-8)
    tensors = lattice.apply2((2, 1), (3, 1), tensors, cnot, 1e-8)
    tensors = lattice.apply2((3, 3), (3, 2), tensors, cnot, 1e-8)
    tensors = lattice.apply2((2, 3), (3, 3), tensors, cnot, 1e-8)
    tensors = lattice.apply2((2, 2), (2, 3), tensors, cnot, 1e-8)
    tensors = lattice.apply2((2, 1), (2, 2), tensors, cnot, 1e-8)
    tensors = lattice.apply2((2, 0), (2, 1), tensors, cnot, 1e-8)
    tensors = lattice.apply2((1, 0), (2, 0), tensors, cnot, 1e-8)
    tensors = lattice.apply2((0, 3), (1, 3), tensors, cnot, 1e-8)
    tensors = lattice.apply2((0, 2), (1, 2), tensors, cnot, 1e-8)
    tensors = lattice.apply2((0, 1), (1, 1), tensors, cnot, 1e-8)
    tensors = lattice.apply2((0, 0), (1, 0), tensors, cnot, 1e-8)
    tensors = lattice.apply2((0, 2), (0, 3), tensors, cnot, 1e-8)
    tensors = lattice.apply2((0, 1), (0, 2), tensors, cnot, 1e-8)
    tensors = lattice.apply2((0, 0), (0, 1), tensors, cnot, 1e-8)
    tensors = lattice.apply1((0, 0), tensors, hadamard)
    message_initializer = get_message_random_nonnegative_initializer(PRNGKey(43))
    messages = lattice.init_messages(message_initializer)
    dist = jnp.finfo(jnp.float64).max
    while dist > 1e-8:
        new_messages = bp_map(tensors, messages)
        dist = messages_frob_distance(new_messages, messages)
        messages = new_messages
    tensors, lmbds = vg_map(tensors, messages)
    tensors, lmbds = lattice.truncate(tensors, lmbds, 1e-8)
    tensors = sg_map(tensors, lmbds)
    message_initializer = get_message_random_nonnegative_initializer(PRNGKey(44))
    messages = lattice.init_messages(message_initializer)
    while dist > 1e-8:
        new_messages = bp_map(tensors, messages)
        dist = messages_frob_distance(new_messages, messages)
        messages = new_messages
    for i in range(4):
        for j in range(4):
            node = lattice.get_node((i, j))
            for d, neighbor in zip(node.bond_shape, node.neighbors):
                assert neighbor.dimension == d, neighbor.dimension
                assert d == 1, d
            dens = get_one_side_density_matrix(node, tensors, messages)
            assert jnp.isclose(
                dens - jnp.array([1, 0, 0, 0], dtype=jnp.complex128).reshape((2, 2)),
                1e-8,
            ).all()
