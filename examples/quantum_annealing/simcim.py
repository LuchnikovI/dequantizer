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
from quantum_annealing.src.simcim import SimCim
from quantum_annealing.src.energy_function import EnergyFunction

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    # extracting the experiment parameters
    seed = int(cfg["task_generator"]["seed"])
    qbp_result_path = Path(cfg["qbp_result_path"])
    sigma = float(cfg["simcim_parameters"]["sigma"])
    attempt_num = int(cfg["simcim_parameters"]["attempt_num"])
    alpha = float(cfg["simcim_parameters"]["alpha"])
    c_th = float(cfg["simcim_parameters"]["c_th"])
    zeta = float(cfg["simcim_parameters"]["zeta"])
    N = int(cfg["simcim_parameters"]["N"])
    dt = float(cfg["simcim_parameters"]["dt"])
    o = float(cfg["simcim_parameters"]["o"])
    d = float(cfg["simcim_parameters"]["d"])
    s = float(cfg["simcim_parameters"]["s"])
    key = PRNGKey(seed)
    key, _ = split(key)

    # logging some of the experiment parameters
    for device in devices():
        log.info(f"Device ID: {device.id}, type: {device.device_kind}")
    log.info(f"Results directory is {qbp_result_path}")
    log.info(f"Random seed is set to {seed}")
    log.info(f"SimCim parameters: {cfg['simcim_parameters']}")

    # lattice uploading
    with h5py.File(qbp_result_path.joinpath("result"), "r") as f:
        fields = np.array(f["fields"])
        pairs = np.array(f["coupled_spin_pairs"])
        coupling_amplitudes = np.array(f["coupling_amplitudes"])
        energy_function = EnergyFunction(
            coupling_amplitudes,
            pairs,
            fields,
        )
    # running SimCim
    log.info(f"SimCim is started")
    simcim = SimCim(
        sigma,
        attempt_num,
        alpha,
        c_th,
        zeta,
        N,
        dt,
        o,
        d,
        s,
        seed,
    )
    simcim_configuration, _ = simcim.evolve(energy_function)
    simcim_configuration = jnp.sign(simcim_configuration)
    simcim_energies = vmap(eval_energy, in_axes=(None, 1))(
        energy_function, simcim_configuration
    )
    simcim_configuration = simcim_configuration[:, jnp.argmin(simcim_energies)]
    log.info(f"SimCim finished")
    log.info(f"Sampled config: {simcim_configuration}")
    simcim_energy = eval_energy(
        energy_function,
        simcim_configuration,
    )
    log.info(f"Sampled energy: {simcim_energy}")
    hdf5_file = h5py.File(f"{qbp_result_path}/simcim_result", "a")
    hdf5_file.create_dataset(
        "coupling_amplitudes", data=energy_function.coupling_amplitudes
    )
    hdf5_file.create_dataset(
        "coupled_spin_pairs", data=energy_function.coupled_spin_pairs
    )
    hdf5_file.create_dataset("fields", data=energy_function.fields)
    hdf5_file.create_dataset(
        "simcim_configuration",
        data=simcim_configuration,
    )
    hdf5_file.create_dataset("simcim_energy", data=simcim_energy)
    hdf5_file.close()


if __name__ == "__main__":
    main()
