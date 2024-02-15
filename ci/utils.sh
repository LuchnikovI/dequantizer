#!/usr/bin/env bash

ci_scripts_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

export DQR_LOG_LEVELS=${LOG_LEVELS:-'DEBUG INFO WARNING ERROR'}

# -------------------------------------------------------------------------------------------

log() {
    local severity=$1
    shift

    local ts=$(date "+%Y-%m-%d %H:%M:%S%z")

    # See https://stackoverflow.com/a/46564084
    if [[ ! " ${DQR_LOG_LEVELS} " =~ .*\ ${severity}\ .* ]] ; then
        log ERROR "Unexpected severity '${severity}', must be one of: ${DQR_LOG_LEVELS}"
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

check_exit_code() {
    if [[ $1 -ne 0 ]]; then
        log ERROR $2
        exit 1
    fi
}