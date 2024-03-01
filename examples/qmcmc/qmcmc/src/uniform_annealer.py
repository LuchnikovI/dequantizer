from typing import Union
import jax.numpy as jnp
from jax import Array
from jax.random import categorical
from .base_annealer import Annealer
from .energy_function import EnergyFunction


class UniformAnnealer(Annealer):
    def __init__(
        self, energy_function: EnergyFunction, flip_probability: Union[Array, float]
    ):
        super().__init__(energy_function)
        self.__flip_probability = flip_probability

    def _transition(self, current_configuration: Array, key: Array) -> Array:
        profile = (
            2
            * categorical(
                key,
                jnp.array(
                    [
                        jnp.log(self.__flip_probability),
                        jnp.log(1 - self.__flip_probability),
                    ]
                ),
                shape=current_configuration.shape,
            )
            - 1
        )
        return current_configuration * profile

    def _transition_probabilities_ratio(
        self,
        current_configuration: Array,
        potential_configuration: Array,
    ) -> Array:
        return jnp.array(1.0)
