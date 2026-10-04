#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${UV_PROJECT_ENVIRONMENT:-$repo_root/.venv}"
python_bin="$venv_dir/bin/python"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is required; install it from https://docs.astral.sh/uv/" >&2
    exit 1
fi

if [[ ! -x "$python_bin" ]] || ! "$python_bin" -c \
    'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
    uv venv --clear --python 3.12 "$venv_dir"
fi

uv pip install --python "$python_bin" numpy

cd "$repo_root"
site_packages="$($python_bin -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
python_libdir="$($python_bin -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"

# PyO3 links standalone test binaries against libpython. On macOS, Python
# installations created by uv may put that dylib outside dyld's default
# search paths; without this export the binary compiles but fails at runtime
# with "Library not loaded: @rpath/libpython..." (or the framework equivalent).
if [[ "$(uname -s)" == "Darwin" && -d "$python_libdir" ]]; then
    export DYLD_LIBRARY_PATH="$python_libdir${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi

PYTHONPATH="$site_packages${PYTHONPATH:+:$PYTHONPATH}" \
    PYO3_PYTHON="$python_bin" \
    cargo test --no-default-features "$@"
