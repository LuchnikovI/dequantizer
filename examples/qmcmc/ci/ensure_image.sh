#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

. "${script_dir}/utils.sh"

# ------------------------------------------------------------------------------------------

if docker image inspect ${QMCMC_IMAGE_NAME} >/dev/null 2>&1; then
    :
else
    log INFO "${QMCMC_IMAGE_NAME} image has not been found"
    . "${script_dir}/build_image.sh"
fi