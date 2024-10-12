#!/usr/bin/env bash

get_help() {
cat << EOF
Runs quantum annealing experiment within a docker container with all preinstalled dependencies. \
If image of the container does not exist, the script builds it first from scratch. \
It mounts the entire home directory inside the container to make the access to files on the machine \
transparent. Note, that this script uses Hydra (https://hydra.cc/docs/intro/) to control configuration
of the experiment and you can override some of the options using Hydra syntax. E.g., to change quantum annealing \
schedule and type of a task that is being solved in the BP based quantum annealing simulation one can run:

./runner.sh qbp task_generator=debug_small_random_regular quantum_annealing_schedule=linear_wo_sampling

Usage: . ${BASH_SOURCE[0]} [COMMAND] [HYDRA_OVERRIDING_OPTIONS]

Commands:

    help: shows this message;

    qbp: runs BP based quantum annealing experiment;

    simcim: runs simcim experiment, one needs also to point to
        a dirrectory with results of BP based quantum annealing simulation as an argument, e.g.
        ./runner.sh simcim +qbp_result_path=<path>;

    exact: runs exact quantum annealing simmulation, one needs also to point to
        a dirrectory with results of BP based quantum annealing simulation as an argument, e.g.
        ./runner.sh exact +qbp_result_path=<path>;

    mqlib: runs MQLib solver, one needs also to point to
        a dirrectory with results of BP based quantum annealing simulation as an argument, e.g.
        ./runner.sh mqlib +qbp_result_path=<path>;

EOF
}

export JAX_ENABLE_X64=True
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
. "${script_dir}/utils.sh"
. "${script_dir}/ensure_image.sh"

if [[ ${QA_USE_CUDA} -eq 1 ]]; then
    cuda_flag="--gpus all"
else
    cuda_flag=""
fi

docker_run="docker run \
    --user $QA_UID:$QA_GID
    --workdir "${script_dir}/.." \
    --mount type=bind,source="${HOME}",target="${HOME}" \
    $cuda_flag \
    ${QA_IMAGE_NAME}"

case $1 in

  help)
        get_help
        exit 0
    ;;
  simcim)
        shift
        ${docker_run} "${script_dir}/../simcim.py" $@
    ;;
  qbp)
        shift
        ${docker_run} "${script_dir}/../qbp.py" $@
    ;;
  exact)
        shift
        ${docker_run} "${script_dir}/../exact.py" $@
    ;;
  mqlib)
        shift
        ${docker_run} "${script_dir}/../mqlib.py" $@
    ;;
  *)
        log ERROR "Unknown command ${1}"
        get_help
        exit 1
    ;;
esac
