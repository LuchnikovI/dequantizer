from typing import List
import copy
import jax.numpy as jnp
from jax import Array
from jax.random import split
from ..graph import (
    get_random_tree_tensor_graph,
    get_message_random_nonnegative_initializer,
    get_tensor_bloated_ghz_initializer,
    Edge,
)
from .belief_propagation import get_belief_propagation_map
from .vidal_gauge import get_vidal_gauge_fixing_map
from .symmetric_gauge import get_symmetric_gauge_fixing_map
from .messages_distance import messages_frob_distance
from .vidal_distance import get_vidal_gauge_distance_map


def random_tree_ghz_gauge_fixing_test(
    nodes_number: int,
    phys_dimension: int,
    bond_dimension: int,
    accuracy: Array,
    key: Array,
):
    tree = get_random_tree_tensor_graph(
        nodes_number,
        phys_dimension,
        bond_dimension,
        key,
    )
    # A map that performs on belief propagation (BP) iteration
    bp_map = get_belief_propagation_map(tree.get_traversal_iterator() or iter([]))
    # A map that fixes the Vidal gauge
    vg_map = get_vidal_gauge_fixing_map(tree.get_traversal_iterator() or iter([]))
    # A map that returns distance between to sets of messages
    vd_map = get_vidal_gauge_distance_map(tree.get_traversal_iterator() or iter([]))
    # A map that fixes the symmetric gauge
    sg_map = get_symmetric_gauge_fixing_map(tree.get_traversal_iterator() or iter([]))
    # Initialization of tensors & messages
    tensors_initializer = get_tensor_bloated_ghz_initializer(key)
    _, key = split(key)
    messages_initializer = get_message_random_nonnegative_initializer(key)
    tensors = tree.init_tensors(tensors_initializer)
    messages = tree.init_messages(messages_initializer)
    # BP until convergence
    dist = jnp.finfo(jnp.float32).max
    while dist > accuracy:
        new_messages = bp_map(tensors, messages)
        dist = messages_frob_distance(new_messages, messages)
        messages = new_messages
    new_tensors, edge_core_tensors = vg_map(tensors, messages)
    # Checking Vidal distance after BP convergence
    assert vd_map(new_tensors, edge_core_tensors) < accuracy
    print("Vidal distance close to zero: OK")
    truncated_tensors, truncated_edge_core_tensors = tree.truncate(
        new_tensors, edge_core_tensors, accuracy
    )
    tensors = sg_map(truncated_tensors, truncated_edge_core_tensors)
    _, key = split(key)
    messages_initializer = get_message_random_nonnegative_initializer(key)
    messages = tree.init_messages(messages_initializer)
    # Checking dimension of truncated edges
    for element in tree.get_traversal_iterator() or iter([]):
        if isinstance(element, Edge):
            assert element.dimension == phys_dimension
    print("Edges dimension after truncation: OK")
    # BP until convergence
    dist = jnp.finfo(jnp.float32).max
    while dist > accuracy:
        new_messages = bp_map(tensors, messages)
        dist = messages_frob_distance(new_messages, messages)
        messages = new_messages
    # Checking messages (all should be proportional identity matrix since the tensor graph is the GHZ state)
    for _, m in messages.items():
        assert (
            jnp.abs(m - jnp.eye(phys_dimension) / jnp.sqrt(float(phys_dimension)))
            < accuracy
        )
    print("All messages ~ I: OK")
