from jax.random import PRNGKey
from .energy_function import random_on_ibm_heavy_hex
from .local_annealer import LocalAnnealer
from .base_annealer import _annealer_runs_smooth_test


def test_local_annealer_run_smooth():
    energy_function = random_on_ibm_heavy_hex(PRNGKey(42))
    local_annealer = LocalAnnealer(energy_function)
    _annealer_runs_smooth_test(local_annealer, PRNGKey(43))
