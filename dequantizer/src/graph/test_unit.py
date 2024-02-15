import jax.numpy as jnp
from jax.random import PRNGKey
from .tensor_initializers import _gen_ghz_core
from .tensor_initializers import get_tensor_bloated_ghz_initializer
from ..graph import TensorGraph


def test_gen_ghz_core():
    for modes_number in range(2, 6):
        for phys_dimension in range(1, 4):
            core = _gen_ghz_core(phys_dimension, modes_number)
            assert jnp.abs(jnp.abs(core).sum() - phys_dimension) < 1e-5
            for i in range(phys_dimension):
                assert jnp.abs(core[modes_number * (i,)] - 1.0) < 1e-5


def test_bloated_ghz_initializer():
    key = PRNGKey(42)
    graph = TensorGraph()
    graph.add_node(3)
    graph.add_node(3)
    graph.add_node(3)
    graph.add_node(3)
    graph.add_edge((0, 1), 5)
    graph.add_edge((1, 2), 4)
    graph.add_edge((1, 3), 6)
    tensors = graph.init_tensors(get_tensor_bloated_ghz_initializer(key))
    result = jnp.tensordot(tensors[0], tensors[1], axes=[0, 0])
    result = jnp.tensordot(tensors[2], result, axes=[0, 1])
    result = jnp.tensordot(tensors[3], result, axes=[0, 2])
    true_result = _gen_ghz_core(3, 4)
    assert result.shape == true_result.shape
    diff = jnp.linalg.norm(result - true_result) / jnp.maximum(
        jnp.linalg.norm(result), jnp.linalg.norm(true_result)
    )
    assert diff < 1e-3
