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
from omegaconf import DictConfig
from quantum_annealing.src.energy_function import (
    random_on_ibm_heavy_hex,
    random_on_one_heavy_hex_loop,
    random_on_small_tree,
    EnergyFunction,
)
from quantum_annealing.src.scheduler import get_scheduler
from quantum_annealing.src.quantum_annealer import run_quantum_annealer
from quantum_annealing.src.classical_annealer import run_classical_annealer
from quantum_annealing.src.energy_evaluation import eval_energy

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    lattice_type = cfg["task_generator"]["lattice_type"]
    field_std = jnp.array(float(cfg["task_generator"]["fields_std"]))
    coupling_std = jnp.array(float(cfg["task_generator"]["couplings_std"]))
    seed = int(cfg["task_generator"]["seed"])
    total_time_step_size = float(
        cfg["quantum_annealing_schedule"]["total_time_step_size"]
    )
    schedule = eval(cfg["quantum_annealing_schedule"]["schedule"])
    quantum_steps_number = int(cfg["quantum_annealing_schedule"]["steps_number"])
    max_chi = int(cfg["emulator_parameters"]["max_chi"])
    max_belief_propagation_iterations = int(
        cfg["emulator_parameters"]["max_belief_propagation_iterations"]
    )
    layers_per_regauging = int(cfg["emulator_parameters"]["layers_per_regauging"])
    accuracy = float(cfg["emulator_parameters"]["accuracy"])
    synchronous_update = bool(cfg["emulator_parameters"]["synchronous_update"])
    traversal_type = cfg["emulator_parameters"]["traversal_type"]
    initial_temperature = cfg["classical_annealer_schedule"]["initial_temperature"]
    final_temperature = cfg["classical_annealer_schedule"]["final_temperature"]
    flip_probability = cfg["classical_annealer_schedule"]["flip_probability"]
    classical_steps_number = cfg["classical_annealer_schedule"]["steps_number"]

    key = PRNGKey(seed)
    key, subkey = split(key)
    output_dir = HydraConfig.get().run.dir

    # logging
    log.info(f"Output directory is {output_dir}")
    log.info(f"Total time step size {total_time_step_size}")
    log.info(f"Quantum annealer steps number {quantum_steps_number}")
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
    log.info(
        f"Initial temperature of a classical annealer is set to {initial_temperature}"
    )
    log.info(f"Final temperature of a classical annealer is set to {final_temperature}")
    log.info(f"Classical annealer number of steps is set to {classical_steps_number}")
    log.info(
        f"Spin flip probability of a classical annealer is set to {flip_probability}"
    )

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

        case other:
            raise NotImplementedError(f"Lattice of type {other} is not implemented.")
    qubits_number = len(energy_function.fields)
    log.info(f"Number of qubits in a lattice is {qubits_number}")
    schedule = get_scheduler(total_time_step_size, schedule, quantum_steps_number)
    key, subkey = split(key)
    classical_annealing_results = run_classical_annealer(
        initial_temperature,
        final_temperature,
        classical_steps_number,
        flip_probability,
        energy_function,
        subkey,
    )
    log.info(f"Quantum annealing is started")
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
    )
    log.info(
        f"Quantum annealing finished, sampled config: {quantum_annealing_results.configuration}"
    )
    quantum_sampled_energy = eval_energy(
        energy_function, quantum_annealing_results.configuration
    )
    log.info(f"Sampled energy: {quantum_sampled_energy}")
    hdf5_file = h5py.File(f"{output_dir}/result", "a")
    hdf5_file.create_dataset(
        "quantum_annealer_configuration", data=quantum_annealing_results.configuration
    )
    hdf5_file.create_dataset("quantum_annealer_energy", data=quantum_sampled_energy)
    hdf5_file.create_dataset(
        "classical_annealer_configuration",
        data=classical_annealing_results.configuration,
    )
    hdf5_file.create_dataset(
        "classical_annealer_energy", data=classical_annealing_results.energy
    )
    hdf5_file.create_dataset(
        "vidal_distances_after_regauging",
        data=quantum_annealing_results.vidal_distances_after_regauging,
    )
    hdf5_file.create_dataset(
        "truncation_affected_vidal_distances",
        data=quantum_annealing_results.truncation_affected_vidal_distances,
    )
    hdf5_file.close()


if __name__ == "__main__":
    main()
