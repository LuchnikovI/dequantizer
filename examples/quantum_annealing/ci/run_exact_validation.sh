#!/usr/bin/env bash

#SBATCH --mail-user=ilia.luchnikov@tii.ae
#SBATCH --job-name=bp_quantum_annealing
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=32GB
#SBATCH --mail-type=END,FAIL

start_seed=1
end_seed=20
start_qubits_number=14
end_qubits_number=26

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
for qubits_number in $(seq $start_qubits_number 2 $end_qubits_number); do
    for seed in {$start_seed..$end_seed}; do
        . "${script_dir}/runner.sh" \
            qbp \
            hydra.run.dir=./outputs/qbp/exact_validation/$qubits_number/\${now:%Y-%m-%d}-\${now:%H-%M-%S} \
            quantum_annealing_schedule=linear_with_history \
            task_generator=exact_validation \
            task_generator.nodes_number=$qubits_number \
            task_generator.seed=$seed
    done
done

for dir in $(find ${script_dir}/../outputs/qbp/exact_validation/ -maxdepth 2 -mindepth 2 -type d); do
    qbp_result_path=./$(realpath -s --relative-to=$script_dir/.. $dir)
    . "${script_dir}/runner.sh" \
        exact \
        +qbp_result_path=./$(realpath -s --relative-to=$script_dir/.. $dir) \
        quantum_annealing_schedule=linear_with_history
done
