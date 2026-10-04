#!/usr/bin/env python3
"""Validate a janitor-rs release tag and built distribution versions.

Cargo.toml is the single version source for the Rust crate and the Maturin
package.  This script checks that source, an optional git tag, and any wheel
or source-distribution metadata all agree before a release is published.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


def cargo_version() -> str:
    """Return the package version reported by Cargo metadata."""
    output = subprocess.check_output(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        text=True,
    )
    metadata = json.loads(output)
    packages = [package for package in metadata["packages"] if package["name"] == "janitor-rs"]
    if len(packages) != 1:
        raise ValueError("expected exactly one janitor-rs package in Cargo metadata")
    return packages[0]["version"]


def metadata_version(contents: str, artifact: Path) -> str:
    """Extract and validate the version field from package metadata."""
    for line in contents.splitlines():
        if line.startswith("Version:"):
            return line.partition(":")[2].strip()
    raise ValueError(f"{artifact} does not contain package version metadata")


def artifact_files(inputs: list[str]) -> list[Path]:
    """Expand artifact files and directories passed by the release workflow."""
    files: list[Path] = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            files.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.suffix == ".whl" or candidate.name.endswith(".tar.gz")
            )
        elif path.suffix == ".whl" or path.name.endswith(".tar.gz"):
            files.append(path)
    return sorted(set(files))


def distribution_version(artifact: Path) -> str:
    """Read the version from a wheel or source distribution."""
    if artifact.suffix == ".whl":
        with zipfile.ZipFile(artifact) as archive:
            metadata_files = [
                name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_files) != 1:
                raise ValueError(f"expected one wheel METADATA file in {artifact}")
            return metadata_version(archive.read(metadata_files[0]).decode(), artifact)

    with tarfile.open(artifact, "r:gz") as archive:
        metadata_files = [
            member for member in archive.getmembers() if member.name.endswith("/PKG-INFO")
        ]
        if len(metadata_files) != 1:
            raise ValueError(f"expected one PKG-INFO file in {artifact}")
        extracted = archive.extractfile(metadata_files[0])
        if extracted is None:
            raise ValueError(f"could not read PKG-INFO from {artifact}")
        return metadata_version(extracted.read().decode(), artifact)


def validate(tag: str | None, artifacts: list[str], require_artifacts: bool = False) -> None:
    """Validate the Cargo version, optional tag, and optional distributions."""
    expected = cargo_version()
    if tag is not None and tag != f"v{expected}":
        raise ValueError(f"tag {tag!r} does not match Cargo version v{expected}")

    files = artifact_files(artifacts)
    if require_artifacts and not files:
        raise ValueError("no wheel or source-distribution artifacts were found")
    for artifact in files:
        actual = distribution_version(artifact)
        if actual != expected:
            raise ValueError(
                f"artifact {artifact} has version {actual}, expected {expected}"
            )

    print(f"release version {expected} validated")
    if files:
        print(f"validated {len(files)} distribution artifact(s)")


def main() -> int:
    """Parse command-line arguments and validate the requested release."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="release tag; must be exactly v< Cargo version >")
    parser.add_argument(
        "--artifacts",
        nargs="*",
        default=[],
        help="wheel or sdist files/directories to validate",
    )
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help="fail when no wheel or source-distribution artifacts are found",
    )
    args = parser.parse_args()
    try:
        validate(args.tag, args.artifacts, args.require_artifacts)
    except (OSError, subprocess.CalledProcessError, ValueError, json.JSONDecodeError) as error:
        print(f"release validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
