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
from jax import vmap
from omegaconf import DictConfig
from quantum_annealing.src.energy_function import (
    random_on_ibm_heavy_hex,
    random_on_one_heavy_hex_loop,
    random_on_small_tree,
    random_regular,
    random_regular_maxcut,
    EnergyFunction,
)
from quantum_annealing.src.scheduler import get_scheduler
from quantum_annealing.src.quantum_annealer import run_quantum_annealer
from quantum_annealing.src.energy_evaluation import eval_energy
from quantum_annealing.src.simcim import SimCim

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    # extracting experiment parameters
    seed = int(cfg["task_generator"]["seed"])
    lattice_type = cfg["task_generator"]["lattice_type"]
    field_std = jnp.array(float(cfg["task_generator"]["fields_std"]))
    coupling_std = jnp.array(float(cfg["task_generator"]["couplings_std"]))
    total_time_step_size = float(
        cfg["quantum_annealing_schedule"]["total_time_step_size"]
    )
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
    # SimCim parameters
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
    key, subkey = split(key)
    output_dir = HydraConfig.get().run.dir

    # logging some experiment parameters
    log.info(f"Output directory is {output_dir}")
    log.info(f"Total time step size {total_time_step_size}")
    log.info(f"Quantum annealer steps number {quantum_steps_number}")
    log.info(f"Quantum annealing sampling flag {sample_measurements}")
    log.info(
        f"Time steps schedule is se by a closure {cfg['quantum_annealing_schedule']['schedule']}"
    )
    log.info(f"Lattice type is set to {lattice_type}")
    log.info(f"Standard deviation of local magnetic fields is set to {field_std}")
    log.info(f"Standard deviation of coupling constants is set to {coupling_std}")
    log.info(f"Random seed is set to {seed}")
    log.info(f"Maximal chi is set to {max_chi}")
    log.info(
        f"Maximal number of belief propagation iterations is ste to {max_belief_propagation_iterations}"
    )
    log.info(f"Number of layers per regauging is set to {layers_per_regauging}")
    log.info(f"Accuracy of belief propagation is set to {accuracy}")
    log.info(f"Synchronous updates are turned to {synchronous_update}")
    log.info(f"Type of graph traversal is set to {traversal_type}")
    log.info(f"SimCim parameters: {cfg['simcim_parameters']}")

    # lattice generation
    energy_function: EnergyFunction
    match lattice_type:

        case "ibm_heavy_hex":
            key, subkey = split(key)
            energy_function = random_on_ibm_heavy_hex(subkey, field_std, coupling_std)

        case "one_heavy_hex_loop":
            key, subkey = split(key)
            energy_function = random_on_one_heavy_hex_loop(
                subkey, field_std, coupling_std
            )

        case "small_tree":
            key, subkey = split(key)
            energy_function = random_on_small_tree(subkey, field_std, coupling_std)

        case "random_regular":
            key, subkey = split(key)
            degree = int(cfg["task_generator"]["degree"])
            nodes_number = int(cfg["task_generator"]["nodes_number"])
            energy_function = random_regular(
                subkey, nodes_number, degree, field_std, coupling_std
            )

        case "random_regular_maxcut":
            key, subkey = split(key)
            degree = int(cfg["task_generator"]["degree"])
            nodes_number = int(cfg["task_generator"]["nodes_number"])
            energy_function = random_regular_maxcut(
                subkey,
                nodes_number,
                degree,
            )

        case other:
            raise NotImplementedError(f"Lattice of type {other} is not implemented.")
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
    log.info(f"SimCim finished, sampled config: {simcim_configuration}")
    simcim_energy = eval_energy(
        energy_function,
        simcim_configuration,
    )
    log.info(f"Sampled energy: {simcim_energy}")
    qubits_number = len(energy_function.fields)
    # running quantum annealing
    log.info(f"Quantum annealing is started")
    log.info(f"Number of qubits in a lattice is {qubits_number}")
    # quantum annealing schedule
    schedule = get_scheduler(total_time_step_size, schedule, quantum_steps_number)
    key, subkey = split(key)
    quantum_annealing_results = run_quantum_annealer(
        energy_function,
        schedule,
        subkey,
        max_chi,
        qubits_number * layers_per_regauging,
        accuracy,
        max_belief_propagation_iterations,
        synchronous_update,
        traversal_type,
        sample_measurements,
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
        "simcim_configuration",
        data=simcim_configuration,
    )
    hdf5_file.create_dataset("simcim_energy", data=simcim_energy)
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
