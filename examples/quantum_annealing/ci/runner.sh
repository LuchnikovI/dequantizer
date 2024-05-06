#!/usr/bin/env bash

get_help() {
cat << EOF
Runs quantum annealing experiment within the container with all preinstalled dependancies. \
If image of the container does not exist, it builds it first. \
It mounts the entire home directory inside the container to make the access to files on the machine \
transparent.

Usage: . ${BASH_SOURCE[0]} [COMMAND] [HYDRA_OPTIONS]

Commands:
    help: shows this message;
    simcim: runs simcim experiment, one needs also to point out
        a dirrectory with results of annealing as an argument, e.g.
        ./runner.sh simcim +qbp_result_path=<path>;
    qbp: runs BP based quantum annealing experiment;
    exact: runs exact quantum annealing simmulation, one needs also to point out
        a dirrectory with results of annealing as an argument, e.g.
        ./runner.sh simcim +qbp_result_path=<path>.

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
  *)
        log ERROR "Unknown command ${1}"
        get_help
        exit 1
    ;;
esac
