## Description

This repository contains Python implementation of graph tensor network canonicalization, contraction, sampling techniques inspired by recent works on belief-propagation applied to graph tensor networks, e.g. [Gauging tensor networks with belief propagation](https://scipost.org/10.21468/SciPostPhys.15.6.222), [Efficient Tensor Network Simulation of IBM’s Eagle Kicked Ising Experiment](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.010308);

It also contains code used to perform numerical experiments for our paper [Large-scale quantum annealing simulation with tensor networks and belief propagation](https://arxiv.org/abs/2409.12240) as one of the examples (see below).

## How to install

To install the accompanying package necessary to run examples first ensure that [Jax](https://github.com/google/jax) (either CPU or GPU version) is installed in your virtual environment. Then clone this repo and run `pip install .` in the root of the cloned repo within your virtual environment.

## How to run examples
Note, that all the examples are quite heavy and may require sufficient computational time. Note also, that all the examples use jax just in time compilation and first iteration in each example could be exhaustively long.
  - `./examples/quantum_annealing/ci/runner.sh` This is the code for [Large-scale quantum annealing simulation with tensor networks and belief propagation](https://arxiv.org/abs/2409.12240) paper where we simulate quantum annealing for large (up to 1000 variables) optimization problems. It has a separate [readme](https://github.com/LuchnikovI/dequantizer/blob/main/examples/quantum_annealing/README.md).
  - `./examples/3d_ising_canonicalization.py` in this example we build a 3d PEPS-like tensor network with bond dimension $2$, that can be seen as a square root of a classical 3D Ising model partition function, i.e. the contraction of the network with itself gives the value of the partition function. We bring it to the Vidal canonical form and calculate average magnetization for different inverse temperatures. This is an example from the [Gauging tensor networks with belief propagation](https://www.scipost.org/SciPostPhys.15.6.222/pdf) (see the last paragraph of section 3.2). This script saves resulting plots in the script directory;
  - `./examples/ibm_eagle_processor.py` In this example we simulate dynamics of 62-nd qubit (a qubit in the middle of a lattice) of the Eagle IBM quantum processor either for 127 qubits (the original IBM processor) or for its infinite modification. Note, that simulation for an infinite processor is way faster due to the translational symmetry. The default set of parameters is aimed on reproducing Fig. 4 e) from [Efficient Tensor Network Simulation of IBM’s Eagle Kicked Ising Experiment](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.010308). This script saves resulting plots in the script directory;
