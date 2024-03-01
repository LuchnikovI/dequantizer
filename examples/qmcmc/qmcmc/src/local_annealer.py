import jax.numpy as jnp
from jax import Array
from jax.random import randint
from .base_annealer import Annealer
from .energy_function import EnergyFunction


class LocalAnnealer(Annealer):
    def __init__(self, energy_function: EnergyFunction):
        super().__init__(energy_function)

    def _transition(self, current_configuration: Array, key: Array) -> Array:
        idx = randint(key, (1,), minval=0, maxval=self.spins_number)
        state = current_configuration[idx]
        return current_configuration.at[idx].set(-state)

    def _transition_probabilities_ratio(
        self,
        current_configuration: Array,
        potential_configuration: Array,
    ) -> Array:
        return jnp.array(1.0)
