import logging
from dataclasses import dataclass
from typing import Union
import jax.numpy as jnp
from jax import Array
from jax.random import randint, categorical, split, uniform
from .energy_evaluation import eval_energy
from .energy_function import EnergyFunction

log = logging.getLogger(__name__)


def _get_proposal(config: Array, flip_probability: Union[float, Array], key: Array):
    profile = (
        2
        * categorical(
            key,
            jnp.array(
                [
                    jnp.log(flip_probability),
                    jnp.log(1 - flip_probability),
                ]
            ),
            shape=config.shape,
        )
        - 1
    )
    return config * profile


@dataclass
class ClassicalAnnealingResults:
    configuration: Array
    energy: Array


def run_classical_annealer(
    initial_temperature: float,
    final_temperature: float,
    steps_number: int,
    flip_probability: float,
    energy_function: EnergyFunction,
    key: Array,
) -> ClassicalAnnealingResults:
    spins_number = energy_function.fields.shape[0]
    decay_coeff = jnp.exp(
        (jnp.log(final_temperature) - jnp.log(initial_temperature)) / steps_number
    )
    temperature = initial_temperature
    config = 2 * randint(key, (spins_number,), 0, 2) - 1
    energy = eval_energy(energy_function, config)
    log.info(
        f"""Classical annealing started. Parameters of annealing:
             initial_temperature: {initial_temperature}
             final_temperature:   {final_temperature}
             steps_number:        {steps_number}
             flip_probability:    {flip_probability}"""
    )
    for _ in range(steps_number):
        log.info(f"Temperature: {temperature}")
        log.info(f"Energy: {energy}")
        key, subkey = split(key)
        new_config = _get_proposal(config, flip_probability, subkey)
        new_energy = eval_energy(energy_function, config)
        key, subkey = split(key)
        sample = uniform(subkey, (1,))
        if jnp.exp(-(new_energy - energy) / temperature) > sample:
            energy = new_energy
            config = new_config
        temperature *= decay_coeff
    return ClassicalAnnealingResults(
        config,
        energy,
    )
