#!/usr/bin/env python3

import os

os.environ["JAX_ENABLE_X64"] = "True"
os.environ["XLA_FLAGS"] = "--xla_backend_optimization_level=0"

import logging
import hydra
from hydra.core.hydra_config import HydraConfig
import h5py  # type: ignore
import jax.numpy as jnp
from jax.random import split, PRNGKey
from jax import devices
from omegaconf import DictConfig
from quantum_annealing.src.scheduler import get_scheduler
from quantum_annealing.src.quantum_annealer import run_quantum_annealer
from quantum_annealing.src.energy_evaluation import eval_energy
from quantum_annealing.src.utils import none_wrap
from quantum_annealing.src.lattice_generator import generate_lattice

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    # extracting experiment parameters
    seed = int(cfg["task_generator"]["seed"])
    lattice_type = cfg["task_generator"]["lattice_type"]
    field_std = jnp.array(float(cfg["task_generator"]["fields_std"]))
    coupling_std = jnp.array(float(cfg["task_generator"]["couplings_std"]))
    degree = none_wrap(int)(cfg["task_generator"].get("degree"))
    nodes_number = none_wrap(int)(cfg["task_generator"].get("nodes_number"))
    total_time_step_size = float(
        cfg["quantum_annealing_schedule"]["total_time_step_size"]
    )
    record_history = bool(cfg["quantum_annealing_schedule"]["record_history"])
    schedule = eval(cfg["quantum_annealing_schedule"]["schedule"])
    quantum_steps_number = int(cfg["quantum_annealing_schedule"]["steps_number"])
    max_chi = int(cfg["emulator_parameters"]["max_chi"])
    sample_measurements = bool(cfg["quantum_annealing_schedule"]["sample_measurements"])
    max_belief_propagation_iterations = int(
        cfg["emulator_parameters"]["max_belief_propagation_iterations"]
    )
    layers_per_regauging = int(cfg["emulator_parameters"]["layers_per_regauging"])
    accuracy = float(cfg["emulator_parameters"]["accuracy"])
    synchronous_update = bool(cfg["emulator_parameters"]["synchronous_update"])
    traversal_type = cfg["emulator_parameters"]["traversal_type"]
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
    log.info(f"Lattice type is set to {lattice_type}")
    log.info(f"Standard deviation of local magnetic fields is set to {field_std}")
    log.info(f"Standard deviation of coupling constants is set to {coupling_std}")
    log.info(f"Random seed is set to {seed}")
    log.info(f"Maximal chi is set to {max_chi}")
    log.info(
        f"Maximal number of belief propagation iterations is set to {max_belief_propagation_iterations}"
    )
    log.info(f"Number of layers per regauging is set to {layers_per_regauging}")
    log.info(f"Accuracy of belief propagation is set to {accuracy}")
    log.info(f"Synchronous updates are turned to {synchronous_update}")
    log.info(f"Type of graph traversal is set to {traversal_type}")

    # lattice generation
    energy_function = generate_lattice(
        lattice_type,
        field_std,
        coupling_std,
        nodes_number,
        degree,
        key,
    )
    qubits_number = len(energy_function.fields)
    # running quantum annealing
    log.info(f"Quantum annealing is started")
    log.info(f"Number of qubits in a lattice is {qubits_number}")
    # quantum annealing schedule
    schedule = get_scheduler(total_time_step_size, schedule, quantum_steps_number)
    key, subkey = split(key)
    edges_number = energy_function.edges_number
    quantum_annealing_results = run_quantum_annealer(
        energy_function,
        schedule,
        subkey,
        max_chi,
        edges_number * layers_per_regauging,
        accuracy,
        max_belief_propagation_iterations,
        synchronous_update,
        traversal_type,
        sample_measurements,
        record_history,
    )
    log.info(f"Quantum annealing finished")
    hdf5_file = h5py.File(f"{output_dir}/result", "a")
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
    hdf5_file.create_dataset(
        "vidal_distances_after_regauging",
        data=quantum_annealing_results.vidal_distances_after_regauging,
    )
    hdf5_file.create_dataset(
        "truncation_affected_vidal_distances",
        data=quantum_annealing_results.truncation_affected_vidal_distances,
    )
    hdf5_file.create_dataset(
        "truncation_errors",
        data=quantum_annealing_results.truncation_errors,
    )
    hdf5_file.create_dataset(
        "entropies",
        data=quantum_annealing_results.entropies,
    )
    hdf5_file.close()


if __name__ == "__main__":
    main()
