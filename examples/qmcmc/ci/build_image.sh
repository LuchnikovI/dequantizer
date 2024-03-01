#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

. "${script_dir}/utils.sh"

# Determine a base image (cuda or cpu based)
if [[ ${QMCMC_USE_CUDA} == 1 ]]; then
    log INFO "Cuda is ON"
    base_image="nvidia/cuda:${QMCMC_CUDA_VERSION}-cudnn${QMCMC_CUDNN_MAJOR_VERSION}-devel-ubuntu${QMCMC_UBUNTU_VERSION}"
    jax_install="--upgrade \"jax[cuda${QMCMC_CUDA_MAJOR_VERSION}_local]\" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
else
    log INFO "Cuda is OFF"
    base_image="ubuntu:${QMCMC_UBUNTU_VERSION}"
    jax_install="--upgrade \"jax[cpu]\""
fi

# ------------------------------------------------------------------------------------------

log INFO "Building an image ${QMCMC_IMAGE_NAME}"
# https://github.com/moby/moby/issues/27393#
docker buildx build --tag "${QMCMC_IMAGE_NAME}" -f - ${script_dir}/../../.. <<EOF

FROM ${base_image}
WORKDIR /dequantizer
COPY . .
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt update && apt install -y software-properties-common pip && \
    pip install --upgrade pip && \
    pip install -U setuptools && \
    pip install numpy pytest -U mypy scipy pylint \
        hydra-core matplotlib black \
        argparse h5py pyyaml ${jax_install} \
        .

EOF

if [[ $? -ne 0 ]];
then
    log ERROR "Failed to build a base image ${QMCMC_IMAGE_NAME}"
    exit 1
else
    log INFO "Base image ${QMCMC_IMAGE_NAME} has been built"
fi
