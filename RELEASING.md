# Releasing janitor-rs

`Cargo.toml` is the single source of truth for the package version.  The
`pyproject.toml` version is dynamic, so Maturin reads the same version when it
builds wheels and source distributions.

## Normal release

The `release-plz` workflow is release-on-demand; ordinary pushes to `main` do
not create a release pull request or publish anything.

When you are ready to release, open the workflow in GitHub Actions, select the
`main` branch, and run it with `command: release-pr`.  Review and merge the
generated pull request, which updates `Cargo.toml` and `CHANGELOG.md`.  Then
run the workflow again with `command: release`.  release-plz creates a GitHub
release and a tag in the form `v<version>`.  The tag starts the Maturin release
workflow, which builds the platform wheels and source distribution, validates
their metadata against Cargo, and publishes them to PyPI.

The release workflow intentionally uses release-plz in git-only mode: janitor-rs
is published as a Python package, not as a Cargo registry package.

## Local validation and manual fallback

Before creating a manual tag, validate the version source:

```console
python scripts/validate_release.py --tag v0.6.2
```

Replace `0.6.2` with the version in `Cargo.toml`.  A manual fallback release
can then be created and pushed with:

```console
git tag v0.6.2
git push origin v0.6.2
```

The tag is required to be exactly `v< Cargo.toml version >`; the release
workflow rejects mismatches before publishing.  A workflow dispatch remains
available for rebuilding the current Cargo version, but the tagged release
path is preferred because it produces an auditable GitHub release.
