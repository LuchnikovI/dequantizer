from jax.random import PRNGKey
from tensor_graph import (
    TensorGraph,
    small_graph_test,
    lattice_3d_test,
    random_tree_test,
)


def test_small_graph():
    key = PRNGKey(42)
    empty_graph = TensorGraph()
    small_graph_test(empty_graph, key)


def test_3d_lattice():
    key = PRNGKey(43)
    lattice_3d_test((4, 5, 6), 4, 3, key)


def test_random_tree():
    key = PRNGKey(44)
    random_tree_test(20, 3, 5, key)
