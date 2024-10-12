#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

. "${script_dir}/utils.sh"


log INFO "Running type checking..."

python3 -m mypy "${script_dir}/.."

if [[ $? -eq 0 ]]; then
    log INFO Type checking: OK
else
    log ERROR Type checking failed
    exit 1
fi

#log INFO "Running tests..."

#python3 -m pytest -vvv "${script_dir}/../quantum_annealing"

#if [[ $? -eq 0 ]]; then
#    log INFO Tests: OK
#else
#    log ERROR Tests failed
#    exit 1
#fi

log INFO "Running linter..."

python3 -m pylint --fail-under=0.0 "${script_dir}/../quantum_annealing"

if [[ $? -eq 0 ]]; then
    log INFO Linting: OK
else
    log ERROR Linting failed
    exit 1
fi

log INFO "Running format checker..."

python3 -m black --check "${script_dir}/.."

if [[ $? -eq 0 ]]; then
    log INFO Formating: OK
else
    log ERROR Format checking failed
    exit 1
fi