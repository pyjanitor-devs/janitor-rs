#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${UV_PROJECT_ENVIRONMENT:-$repo_root/.venv}"
python_bin="$venv_dir/bin/python"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is required; install it from https://docs.astral.sh/uv/" >&2
    exit 1
fi

if [[ ! -x "$python_bin" ]]; then
    uv venv --python 3.12 "$venv_dir"
fi

uv pip install --python "$python_bin" numpy

cd "$repo_root"
site_packages="$($python_bin -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
PYTHONPATH="$site_packages${PYTHONPATH:+:$PYTHONPATH}" \
    PYO3_PYTHON="$python_bin" \
    cargo test --no-default-features "$@"
