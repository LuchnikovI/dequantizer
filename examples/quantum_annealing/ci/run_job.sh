#!/usr/bin/env bash

#SBATCH --mail-user=ilia.luchnikov@tii.ae
#SBATCH --job-name=bp_quantum_annealing
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=128GB
#SBATCH --mail-type=END,FAIL

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

. "${script_dir}/runner.sh" \
    task_generator=very_large_random_regular

. "${script_dir}/runner.sh" \
    task_generator=large_random_regular_maxcut \
    quantum_annealing_schedule=linear_with_sampling_long

. "${script_dir}/runner.sh" \
    task_generator=large_random_regular

. "${script_dir}/runner.sh" \
    task_generator=medium_random_regular_maxcut \
    quantum_annealing_schedule=linear_with_sampling_long

. "${script_dir}/runner.sh" \
    task_generator=medium_random_regular \
    quantum_annealing_schedule=linear_with_sampling

. "${script_dir}/runner.sh" \
    task_generator=standard_ibm_heavy_hex \
    quantum_annealing_schedule=linear_with_sampling

# here we collect some statistics --------------------------------------------------

for i in {1..10}
do
    . "${script_dir}/runner.sh" \
        task_generator.seed=$i
done