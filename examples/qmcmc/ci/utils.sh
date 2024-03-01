#!/usr/bin/env bash

ci_scripts_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# has python 3.10 by default
export QMCMC_UBUNTU_VERSION=${QMCMC_UBUNTU_VERSION:-"22.04"}
export QMCMC_USE_CUDA=${QMCMC_USE_CUDA:-0}
export QMCMC_LOG_LEVELS=${QMCMC_LOG_LEVELS:-'DEBUG INFO WARNING ERROR'}

# see https://github.com/google/jax for more information
export QMCMC_CUDA_VERSION=${QMCMC_CUDA_VERSION:-'11.8.0'}
export QMCMC_CUDNN_MAJOR_VERSION=${QMCMC_CUDNN_MAJOR_VERSION:-'8'}

export QMCMC_CUDA_MAJOR_VERSION="$(echo ${QMCMC_CUDA_VERSION} | grep -oP "[0-9]+" | head -1)"

# checking cuda flag
if [[ ${QMCMC_USE_CUDA} == 1 ]]; then
    export QMCMC_IMAGE_NAME="qmcmc.cuda"
else
    export QMCMC_IMAGE_NAME="qmcmc.cpu"
fi

# -------------------------------------------------------------------------------------------

log() {
    local severity=$1
    shift

    local ts=$(date "+%Y-%m-%d %H:%M:%S%z")

    # See https://stackoverflow.com/a/46564084
    if [[ ! " ${QMCMC_LOG_LEVELS} " =~ .*\ ${severity}\ .* ]] ; then
        log ERROR "Unexpected severity '${severity}', must be one of: ${QMCMC_LOG_LEVELS}"
        severity=ERROR
    fi

    # See https://stackoverflow.com/a/29040711 and https://unix.stackexchange.com/a/134219
    local module=$(caller | awk '
        function basename(file, a, n) {
            n = split(file, a, "/")
            return a[n]
        }
        { printf("%s:%s\n", basename($2), $1) }')

    case "${severity}" in
        ERROR)
            color_start='\033[0;31m' # Red
            ;;
        WARNING)
            color_start='\033[1;33m' # Yellow
            ;;
        INFO)
            color_start='\033[1;32m' # Light Green
            ;;
        DEBUG)
            color_start='\033[0;34m' # Blue
            ;;
    esac
    color_end='\033[0m'

    printf "# ${ts} ${color_start}${severity}${color_end} [${module}]: ${color_start}$*${color_end}\n" >&2
}
