#!/usr/bin/env python3

import os

os.environ["JAX_ENABLE_X64"] = "True"
os.environ["XLA_FLAGS"] = "--xla_backend_optimization_level=0"

import logging
from pathlib import Path
import numpy as np
import hydra
from hydra.core.hydra_config import HydraConfig
import h5py  # type: ignore
import jax.numpy as jnp
from jax.random import split, PRNGKey
from jax import devices
from omegaconf import DictConfig
from quantum_annealing.src.scheduler import get_scheduler
from quantum_annealing.src.exact_quantum_annealer import run_exact_quantum_annealer
from quantum_annealing.src.energy_evaluation import eval_energy
from quantum_annealing.src.energy_function import EnergyFunction

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    # extracting experiment parameters
    seed = int(cfg["task_generator"]["seed"])
    qbp_result_path = Path(cfg["qbp_result_path"])
    total_time_step_size = float(
        cfg["quantum_annealing_schedule"]["total_time_step_size"]
    )
    record_history = bool(cfg["quantum_annealing_schedule"]["record_history"])
    schedule = eval(cfg["quantum_annealing_schedule"]["schedule"])
    quantum_steps_number = int(cfg["quantum_annealing_schedule"]["steps_number"])
    sample_measurements = bool(cfg["quantum_annealing_schedule"]["sample_measurements"])
    key = PRNGKey(seed)
    key, subkey = split(key)
    output_dir = HydraConfig.get().run.dir

    # logging some experiment parameters
    for device in devices():
        log.info(f"Device ID: {device.id}, type: {device.device_kind}")
    log.info(f"Output directory is {output_dir}")
    log.info(f"Total time step size {total_time_step_size}")
    log.info(f"Quantum annealer steps number {quantum_steps_number}")
    log.info(f"Quantum annealing sampling flag {sample_measurements}")
    log.info(f"Record history flag {record_history}")
    log.info(
        f"Time steps schedule is se by a closure {cfg['quantum_annealing_schedule']['schedule']}"
    )
    log.info(f"Random seed is set to {seed}")

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
    qubits_number = energy_function.nodes_number
    # running quantum annealing
    log.info(f"Exact quantum annealing is started")
    log.info(f"Number of qubits in a lattice is {qubits_number}")
    # quantum annealing schedule
    schedule = get_scheduler(total_time_step_size, schedule, quantum_steps_number)
    key, subkey = split(key)
    quantum_annealing_results = run_exact_quantum_annealer(
        energy_function,
        schedule,
        subkey,
        sample_measurements,
        record_history,
    )
    log.info(f"Quantum annealing finished")
    hdf5_file = h5py.File(f"{qbp_result_path}/exact_result", "a")
    if quantum_annealing_results.configuration is not None:
        log.info(f"Sampled config: {quantum_annealing_results.configuration}")
        quantum_sampled_energy = eval_energy(
            energy_function, quantum_annealing_results.configuration
        )
        log.info(f"Sampled energy: {quantum_sampled_energy}")
        hdf5_file.create_dataset(
            "quantum_annealer_configuration",
            data=quantum_annealing_results.configuration,
        )
        hdf5_file.create_dataset("quantum_annealer_energy", data=quantum_sampled_energy)
    if record_history:
        hdf5_file.create_dataset(
            "density_matrices_history",
            data=jnp.array(quantum_annealing_results.density_matrices_history),
        )
    quantum_energy_wo_sampling = eval_energy(
        energy_function,
        2 * (quantum_annealing_results.density_matrices[:, 0, 0] > 0.5) - 1,
    )
    log.info(f"Energy without sampling: {quantum_energy_wo_sampling}")
    hdf5_file.create_dataset(
        "coupling_amplitudes", data=energy_function.coupling_amplitudes
    )
    hdf5_file.create_dataset(
        "coupled_spin_pairs", data=energy_function.coupled_spin_pairs
    )
    hdf5_file.create_dataset("fields", data=energy_function.fields)
    hdf5_file.create_dataset(
        "density_matrices",
        data=quantum_annealing_results.density_matrices,
    )
    hdf5_file.create_dataset(
        "quantum_annealer_energy_wo_sampling", data=quantum_energy_wo_sampling
    )
    hdf5_file.close()


if __name__ == "__main__":
    main()
