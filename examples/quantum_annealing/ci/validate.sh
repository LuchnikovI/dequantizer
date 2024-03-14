#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

. "${script_dir}/utils.sh"
. "${script_dir}/ensure_image.sh"

if [[ ${QA_USE_CUDA} -eq 1 ]]; then
    cuda_flag="--gpus all"
else
    cuda_flag=""
fi

docker run \
    --mount type=bind,source="${HOME}",target="${HOME}" \
    $cuda_flag \
    ${QA_IMAGE_NAME} /bin/bash -c "${script_dir}/validation_script.sh"