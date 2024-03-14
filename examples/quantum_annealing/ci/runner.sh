#!/usr/bin/env bash

get_help() {
cat << EOF
Runs scripts within the container with all preinstalled dependancies. \
If image of the container does not exist, it builds it first. \
It mounts the entire home directory to make the access to files on the machine \
transparent.

Usage: . ${BASH_SOURCE[0]} [OPTION] SCRIPT

Options:
    --help: shows this message and ignores the command;
Scripts:
    compare_optimization: a script that runs annealers convergence comparison (see root of qmcmc example)

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
    --workdir "${script_dir}/.." \
    --mount type=bind,source="${HOME}",target="${HOME}" \
    $cuda_flag \
    ${QA_IMAGE_NAME}"

case $1 in

  --help)
        get_help
        exit 0
    ;;
  compare_optimization)
        shift
        ${docker_run} "${script_dir}/../compare_optimization.py" $@
    ;;
  *)
        echo "Unknown command: '$1'"
        get_help
        exit 1
    ;;
esac
