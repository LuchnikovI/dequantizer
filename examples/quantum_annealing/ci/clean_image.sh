#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

. "${script_dir}/utils.sh"

log INFO Cleaning ${QA_IMAGE_NAME} image...

id=$(docker images -q ${QA_IMAGE_NAME})

if [[ -z $id ]]; then
    log INFO ${QA_IMAGE_NAME} does not exist
    exit 1
fi

if docker image rmi $id; then
    log INFO ${QA_IMAGE_NAME} image is cleaned
else
    log ERROR Failed to clean ${QA_IMAGE_NAME} image
    exit 1
fi