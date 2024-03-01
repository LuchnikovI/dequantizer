#!/usr/bin/env python3

import logging
import jax.numpy as jnp
import h5py  # type: ignore
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from jax.random import PRNGKey, split
from qmcmc.src.energy_function import random_on_ibm_heavy_hex, EnergyFunction
from qmcmc.src.base_annealer import Annealer as BaseAnnealer
from qmcmc.src.local_annealer import LocalAnnealer
from qmcmc.src.uniform_annealer import UniformAnnealer

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):

    # parameters from a config
    initial_temperature = jnp.array(
        float(cfg["temperature_vs_iter"]["initial_temperature"])
    )
    final_temperature = jnp.array(
        float(cfg["temperature_vs_iter"]["final_temperature"])
    )
    temperature_schedule = eval(cfg["temperature_vs_iter"]["temperature_schedule"])
    lattice_type = cfg["task_generator"]["lattice_type"]
    field_std = jnp.array(float(cfg["task_generator"]["fields_std"]))
    coupling_std = jnp.array(float(cfg["task_generator"]["couplings_std"]))
    seed = int(cfg["task_generator"]["seed"])
    annealers_params = cfg["annealers"]
    key = PRNGKey(seed)
    output_dir = HydraConfig.get().run.dir

    # logging
    log.info(f"Output directory is {output_dir}")
    log.info(f"Initial temperature is set to {initial_temperature}")
    log.info(f"Final temperature is set to {final_temperature}")
    log.info(
        f"Temperature schedule is set by the closure {cfg['temperature_vs_iter']['temperature_schedule']}"
    )
    log.info(f"Lattice type is set to {lattice_type}")
    log.info(f"Standard deviation of local magnetic fields is set to {field_std}")
    log.info(f"Standard deviation of coupling constants is set to {coupling_std}")
    log.info(f"Random seed is set to {seed}")

    # lattice generation
    energy_function: EnergyFunction
    match lattice_type:

        case "ibm_heavy_hex":
            key, subkey = split(key)
            energy_function = random_on_ibm_heavy_hex(subkey, field_std, coupling_std)

        case other:
            raise NotImplementedError(f"Lattice of type {other} is not implemented.")

    # iterate over annealers
    annealer: BaseAnnealer
    for annealer_params in annealers_params:

        # choosing annealer
        match annealer_params["id"]:

            case "local":
                log.info("Starting LOCAL annealer")
                annealer = LocalAnnealer(energy_function)

            case "uniform":
                log.info(
                    f"Starting UNIFORM annealer, flip probability is {annealer_params['flip_probability']}"
                )
                annealer = UniformAnnealer(
                    energy_function, float(annealer_params["flip_probability"])
                )

            case "quantum":
                log.info("Starting QUANTUM annealer")
                raise NotImplementedError()

            case other:
                raise NotImplementedError(
                    f"Annealer of type {other} is not implemented."
                )

        # running annealing
        key, subkey = split(key)
        result = annealer.run(
            initial_temperature,
            final_temperature,
            temperature_schedule,
            lambda _, energy: log.info(f"Energy value: {energy}"),
            subkey,
        )
        hdf5_file = h5py.File(
            f"{output_dir}/{annealer_params['id']}_annealer_result", "a"
        )
        result.to_hdf5(hdf5_file)
        hdf5_file.close()


if __name__ == "__main__":
    main()
