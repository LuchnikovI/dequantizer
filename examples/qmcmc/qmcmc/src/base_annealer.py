from dataclasses import dataclass
from typing import Union, Callable, Optional, List
from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax import Array
from jax.random import randint, split, uniform
from h5py import File  # type: ignore
from .invariants import (
    _check_couplings_consistency,
    _check_spins_number,
    _check_coupled_spin_pairs,
)
from .energy_function import EnergyFunction
from .tensorop import _eval_energy

"""Annealing results.
Fields:
    spins_configuration_vs_iteration: a matrix whose entries are spin states,
        zeroth index enumerate discrete time, first index enumerate spins;
    energy_vs_iteration: a one dimensional array showing how energy depends on the
        iteration number;
    temperature_vs_iteration: a one-dimensional array showing how temperature depends on the
        iteration number;
    acceptance_vs_iteration: a one-dimensional array showing if for the given number of iteration
        update was accepted (true) or rejected (false)."""


@dataclass
class AnnealingResults:
    spins_configuration_vs_iteration: List[Array]
    energy_vs_iteration: List[Array]
    temperature_vs_iteration: List[Array]
    acceptance_vs_iteration: List[Array]

    def to_hdf5(self, hdf5_file: File):
        hdf5_file.create_dataset(
            "spins_configuration_vs_iteration",
            data=jnp.array(self.spins_configuration_vs_iteration),
        )
        hdf5_file.create_dataset(
            "energy_vs_iteration", data=jnp.array(self.energy_vs_iteration)
        )
        hdf5_file.create_dataset(
            "temperature_vs_iteration", data=jnp.array(self.temperature_vs_iteration)
        )
        hdf5_file.create_dataset(
            "acceptance_vs_iteration", data=jnp.array(self.acceptance_vs_iteration)
        )


"""A base class for annealers."""


class Annealer(ABC):
    """Creates an instance of an annealer.
    Args:
        energy_function: an object representing energy function."""

    def __init__(
        self,
        energy_function: EnergyFunction,
    ):
        _check_coupled_spin_pairs(energy_function.coupled_spin_pairs)
        _check_couplings_consistency(
            energy_function.coupling_amplitudes, energy_function.coupled_spin_pairs
        )
        _check_spins_number(energy_function.coupled_spin_pairs, energy_function.fields)
        self.__coupling_amplitudes = energy_function.coupling_amplitudes
        self.__coupled_spin_pairs = energy_function.coupled_spin_pairs
        self.__fields = energy_function.fields
        self.__spins_number = energy_function.fields.shape[0]

    """Returns number of spins."""

    @property
    def spins_number(self) -> int:
        return self.__spins_number

    @abstractmethod
    def _transition(self, current_configuration: Array, key: Array) -> Array:
        pass

    @abstractmethod
    def _transition_probabilities_ratio(
        self,
        current_configuration: Array,
        potential_configuration: Array,
    ) -> Array:
        pass

    """Runs annealing.
    Args:
        initial_temperature: initial annealing temperature;
        final_temperature: final annealing temperature;
        temperature_schedule: a function that takes previous temperature, iteration number and
            returns new temperature;
        callback: a function that is called at each iteration and takes current spins
            configuration and current energy as an input;
        key: jax random seed.
    Returns:
        AnnealingResults dataclass containing results of annealing."""

    def run(
        self,
        initial_temperature: Union[float, Array],
        final_temperature: Union[float, Array],
        temperature_schedule: Callable[[Union[float, Array], int], Union[float, Array]],
        callback: Optional[Callable[[Array, Union[float, Array]], None]],
        key: Array,
    ) -> AnnealingResults:
        result = AnnealingResults([], [], [], [])
        it = 0
        temperature = initial_temperature
        key, subkey = split(key)
        config = 2 * randint(subkey, (self.__spins_number,), 0, 2) - 1
        energy = _eval_energy(
            config,
            self.__coupling_amplitudes,
            self.__coupled_spin_pairs,
            self.__fields,
        )
        while temperature >= final_temperature:
            if callback is not None:
                callback(config, energy)
            result.temperature_vs_iteration.append(jnp.array(temperature))
            result.spins_configuration_vs_iteration.append(config)
            result.energy_vs_iteration.append(energy)
            key, subkey = split(key)
            new_config = self._transition(config, subkey)
            if not jnp.logical_or(new_config == 1, new_config == -1).all():
                raise ValueError(
                    "Bad proposal of a new config, all spins must take either 1 or -1 value."
                )
            new_energy = _eval_energy(
                new_config,
                self.__coupling_amplitudes,
                self.__coupled_spin_pairs,
                self.__fields,
            )
            transition_probabilities_ratio = self._transition_probabilities_ratio(
                config, new_config
            )
            prob = (
                jnp.exp((energy - new_energy) / temperature)
                * transition_probabilities_ratio
            )
            key, subkey = split(key)
            sample = uniform(subkey, (1,))
            if sample < prob:
                config = new_config
                energy = new_energy
                result.acceptance_vs_iteration.append(jnp.array(True))
            else:
                result.acceptance_vs_iteration.append(jnp.array(False))
            temperature = temperature_schedule(temperature, it)
            it += 1
        return result


def _annealer_runs_smooth_test(annealer: Annealer, key: Array):
    energy_vs_iteration = []
    config_vs_iteration = []

    def callback(conf: Array, energy: Union[float, Array]):
        energy_vs_iteration.append(energy)
        config_vs_iteration.append(conf)

    results = annealer.run(
        10.0,
        0.01,
        lambda t, i: t / (i + 2),
        callback,
        key,
    )
    iter = 1
    temp = 10.0
    correct_temperature_schedule = []
    while temp >= 0.01:
        correct_temperature_schedule.append(temp)
        temp = temp / (iter + 1)
        iter += 1
    correct_temperature_schedule_arr = jnp.array(correct_temperature_schedule)
    assert jnp.isclose(
        jnp.array(results.temperature_vs_iteration), correct_temperature_schedule_arr
    ).all(), f"{jnp.array(results.temperature_vs_iteration)}, {correct_temperature_schedule_arr}"
    assert jnp.isclose(
        jnp.array(results.energy_vs_iteration), jnp.array(energy_vs_iteration)
    ).all(), (
        f"{jnp.array(results.energy_vs_iteration)}, {jnp.array(energy_vs_iteration)}"
    )
    assert jnp.isclose(
        jnp.array(results.spins_configuration_vs_iteration),
        jnp.array(config_vs_iteration),
    ).all(), f"{jnp.array(results.spins_configuration_vs_iteration)}, {jnp.array(config_vs_iteration)}"
