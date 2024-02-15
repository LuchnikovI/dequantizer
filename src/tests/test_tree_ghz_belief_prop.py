from jax.random import PRNGKey
from ..mappings.utils_testing import random_tree_ghz_gauge_fixing_test


def test_random_tree_ghz_gauge_fixing():
    random_tree_ghz_gauge_fixing_test(
        20,
        3,
        [3, 4, 5],
        1e-8,
        PRNGKey(42),
    )
