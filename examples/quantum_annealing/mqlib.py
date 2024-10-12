#!/usr/bin/env python3

import logging
from pathlib import Path
import hydra
import numpy as np
import h5py  # type: ignore
import jax.numpy as jnp
from jax import vmap, devices
from jax.random import split, PRNGKey
from omegaconf import DictConfig
from quantum_annealing.src.energy_evaluation import eval_energy
from quantum_annealing.src.mqlib import MQLib
from quantum_annealing.src.energy_function import EnergyFunction

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    # extracting the experiment parameters
    seed = int(cfg["task_generator"]["seed"])
    qbp_result_path = Path(cfg["qbp_result_path"])
    time_limit = int(cfg["mqlib_parameters"]["time_limit"])
    heuristics = cfg["mqlib_parameters"]["heuristics"]

    # logging some of the experiment parameters
    for device in devices():
        log.info(f"Device ID: {device.id}, type: {device.device_kind}")
    log.info(f"Results directory is {qbp_result_path}")
    log.info(f"Random seed is set to {seed}")
    log.info(f"MQLib parameters: {cfg['mqlib_parameters']}")

    # lattice uploading
    with h5py.File(qbp_result_path.joinpath("result"), "r") as f:
        fields = np.array(f["fields"])
        pairs = np.array(f["coupled_spin_pairs"])
        coupling_amplitudes = np.array(f["coupling_amplitudes"])
        energy_function = EnergyFunction(
            jnp.array(coupling_amplitudes),
            jnp.array(pairs),
            jnp.array(fields),
        )
    hdf5_file = h5py.File(f"{qbp_result_path}/mqlib_result", "a")
    hdf5_file.create_dataset(
        "coupling_amplitudes", data=energy_function.coupling_amplitudes
    )
    hdf5_file.create_dataset(
        "coupled_spin_pairs", data=energy_function.coupled_spin_pairs
    )
    hdf5_file.create_dataset("fields", data=energy_function.fields)
    # running MQLib
    for heuristic in heuristics:
        log.info(f"MQLib for heuristic {heuristic} is started")
        mqlib = MQLib(
            time_limit,
            seed,
            heuristic,
        )
        result = mqlib.run(energy_function)
        log.info(f"MQLib for heuristic {heuristic} finished")
        log.info(f"Sampled config: {result.configuration}")
        log.info(f"MQLib evaluated energy: {result.energy}")
        energy = eval_energy(
            energy_function,
            result.configuration,
        )
        log.info(f"Script evaluated energy: {energy}")
        hdf5_file.create_dataset(
            f"{heuristic}_configuration",
            data=result.configuration,
        )
        hdf5_file.create_dataset(f"{heuristic}_energy", data=energy)
    hdf5_file.close()


if __name__ == "__main__":
    main()
