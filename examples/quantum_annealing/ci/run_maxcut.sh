#!/usr/bin/env bash

#SBATCH --mail-user=ilia.luchnikov@tii.ae
#SBATCH --job-name=bp_quantum_annealing
#SBATCH --nodes=1
#SBATCH --time=4-00:00:00
#SBATCH --mem=128GB
#SBATCH --mail-type=END,FAIL

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

. "${script_dir}/runner.sh" \
    qbp \
    task_generator=medium_random_regular_maxcut \
    quantum_annealing_schedule=linear_with_sampling_long \
    emulator_parameters=large_chi
