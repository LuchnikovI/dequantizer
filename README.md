## Description

This repo contains Python implementations of:
   - gauging technique from [Gauging tensor networks with belief propagation](https://www.scipost.org/SciPostPhys.15.6.222/pdf) paper;
   - QPU simulation technique from [Efficient Tensor Network Simulation of IBM’s Eagle Kicked Ising Experiment](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.010308) paper;
   - belief propagation based sampling of quantum measurements.

## How to install

To install the accompanying package necessary to run examples first ensure that [Jax](https://github.com/google/jax) (either CPU or GPU version) is installed in your virtual environment. Then clone this repo and run `pip install .` in the root of the cloned repo within your virtual environment.

## How to run examples
Note, that all the examples are quite heavy and may require sufficient computational time. Note also, that all the examples use jax just in time compilation and first iteration in each example could be exhaustively long.
After installation of the accompanying package you can run number of examples:
  - `./examples/3d_ising_canonicalization.py` in this example we build a 3d PEPS-like tensor network with bond dimension $2$, that can be seen as a square root of a classical 3D Ising model partition function, i.e. the contraction of the network with itself gives the value of the partition function. We bring it to the Vidal canonical form and calculate average magnetization for different inverse temperatures. This is an example from the [Gauging tensor networks with belief propagation](https://www.scipost.org/SciPostPhys.15.6.222/pdf) (see the last paragraph of section 3.2). This script saves resulting plots in the script directory;
  - `./examples/ibm_eagle_processor.py` In this example we simulate dynamics of 62-nd qubit (a qubit in the middle of a lattice) of the Eagle IBM quantum processor either for 127 qubits (the original IBM processor) or for its infinite modification. Note, that simulation for an infinite processor is way faster due to the translational symmetry. The default set of parameters is aimed on reproducing Fig. 4 e) from [Efficient Tensor Network Simulation of IBM’s Eagle Kicked Ising Experiment](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.010308). This script saves resulting plots in the script directory;
  - `./examples/quantum_annealing/ci/runner.sh` see [extended description](https://github.com/LuchnikovI/dequantizer/blob/main/examples/quantum_annealing/README.md).
