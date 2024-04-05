## Description

This example aimed at quantum annealing simulation run on IBM [Heavy Hex lattice](https://www.nature.com/articles/s41586-023-06096-3) composed of 127 qubits. It adiabatically transforms Hamiltonian from a trivial transverse field Hamiltonian $\sum_i \sigma_x^i$ to an Ising Hamiltonian $\sum_{\langle i, j\rangle} J_{ij}\sigma^i_z\sigma^j_z$ whose coupling constants are randomly sampled and $\langle i,j \rangle$ denotes neighboring spins on the Heavy Hex lattice. After that, one performs quantum measurements in the computational basis obtaining a spin configuration with close to the minimal energy of the Ising Hamiltonian.

## Experiment configuration
One can see and tune an experiment parameters modifying a hierarchical config `./configs` (see documentation of [Hydra](https://hydra.cc/) for more details on the config structure).

## Results
With default parameter from this repo one can closely resemble real quantum annealing process run on the IBM Heavy Hex lattice. The larges error is introduced at the measurements sampling stage, since belief propagation after each measurement does not fully converge. The average deviation of the Vidal distance after each regauging step during the sampling stage is about $0.02$, therefor, one can consider measurements error to be around $0.02$. The more accurate error analysis and comparison of the found suboptimal ground state energy with combinatorial solvers will be done soon. Results of each experiment run (`*.hdf` file and logs) are saved in the `./output` directory with an appropriate time stamp.

## How to run
The experiment is run inside a [Docker](https://www.docker.com/) container, therefore one needs to have only Docker installed on your machine. To run the experiment execute the `./ci/runner.sh` command from the `quantum_annealing` directory.
