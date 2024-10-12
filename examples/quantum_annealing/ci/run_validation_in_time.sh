#!/usr/bin/env bash

#SBATCH --mail-user=ilia.luchnikov@tii.ae
#SBATCH --job-name=bp_quantum_annealing
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=32GB
#SBATCH --mail-type=END,FAIL

start_seed=1
end_seed=20
start_time=100
end_time=800
time_incr=100
qubits_number=24

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
for time in $(seq $start_time $time_incr $end_time); do
    for seed in $(seq $start_seed 1 $end_seed); do
	echo "Current seed: $seed"
	echo "Current time: $time"
	result_subdir=$time/$(date +"%m-%d-%Y-%H-%M-%S")
        . "${script_dir}/runner.sh" \
            qbp \
            hydra.run.dir=./outputs/qbp/exact_validation_in_time/$result_subdir \
            quantum_annealing_schedule=linear_with_history \
            quantum_annealing_schedule.steps_number=$time \
	    task_generator=exact_validation \
	    task_generator.nodes_number=$qubits_number \
            task_generator.seed=$seed

	full_result_path="${script_dir}/../outputs/qbp/exact_validation_in_time/${result_subdir}"	
	rel_result_path=./$(realpath -s --relative-to=$script_dir/.. $full_result_path)
	. "${script_dir}/runner.sh" \
	    exact \
	    +qbp_result_path=$rel_result_path \
	    quantum_annealing_schedule=linear_with_history \
	    quantum_annealing_schedule.steps_number=$time \
	    task_generator=exact_validation \
	    task_generator.nodes_number=$qubits_number \
	    task_generator.seed=$seed
    done
done
