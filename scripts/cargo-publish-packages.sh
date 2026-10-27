#!/bin/bash
#
# Publish every package in this workspace to crates.io in dependency order.
#
# Run this from the repository root after committing all changes: cargo publish
# refuses to publish from a dirty working tree.
#
# crates.io indexes a new version asynchronously, so after each successful
# upload this script polls the crates.io API until the version is visible before
# publishing the packages that depend on it. If a dependent package still fails
# with a "no matching package/version" error, simply re-run the script: it skips
# packages that are already published and resumes where it stopped.
#
# Tunables (environment variables):
#   WAIT_ATTEMPTS  number of index polls per package (default: 60)
#   WAIT_INTERVAL  seconds between polls (default: 10)

set -euo pipefail

# The dependency order is determined by the internal dependencies between crates.
#
# 1. irmf-slicer (core library, no internal dependencies)
# 2. irmf-include-resolver (utility library, no internal dependencies)
# 3. irmf-output-voxels (depends on irmf-slicer)
# 4. irmf-output-stl (depends on irmf-output-voxels and irmf-slicer)
# 5. volume-to-irmf (depends on irmf-slicer)
# 6. irmf-linter (depends on irmf-slicer and irmf-include-resolver)
# 7. compress-irmf (depends on irmf-slicer)
# 8. irmf-slicer-cli (depends on irmf-slicer, irmf-include-resolver, irmf-output-stl, irmf-output-voxels)
# 9. dlp-to-irmf (depends on volume-to-irmf)
# 10. binvox-to-irmf (depends on volume-to-irmf)
# 11. svx-to-irmf (depends on volume-to-irmf)
# 12. stl-to-irmf (depends on volume-to-irmf)
# 13. obj-to-irmf (depends on volume-to-irmf)
# 14. zip-to-irmf (depends on volume-to-irmf)
# 15. cal-optimize (depends on irmf-slicer)
# 16. cal-hardware (depends on cal-optimize)
# 17. irmf-cal-cli (depends on cal-optimize, cal-hardware, and irmf-slicer)

packages=(
    "irmf-slicer"
    "irmf-include-resolver"
    "irmf-output-voxels"
    "irmf-output-stl"
    "volume-to-irmf"
    "irmf-linter"
    "compress-irmf"
    "irmf-slicer-cli"
    "dlp-to-irmf"
    "binvox-to-irmf"
    "svx-to-irmf"
    "stl-to-irmf"
    "obj-to-irmf"
    "zip-to-irmf"
    "cal-optimize"
    "cal-hardware"
    "irmf-cal-cli"
)

WAIT_ATTEMPTS=${WAIT_ATTEMPTS:-60}
WAIT_INTERVAL=${WAIT_INTERVAL:-10}

require_tool() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "error: $1 is required but was not found on PATH." >&2
        exit 1
    fi
}

package_version() {
    (cd "$1" && cargo pkgid | sed 's/.*#//')
}

# Returns success when the exact version is listed for the crate on crates.io.
is_version_published() {
    local pkg=$1
    local version=$2

    # crates.io returns HTTP 403 to requests without a descriptive User-Agent.
    curl -sS --fail --max-time 10 \
        -A "irmf-cargo-publish-packages/1.0 (+https://github.com/gmlewis/rust-irmf-slicer)" \
        "https://crates.io/api/v1/crates/${pkg}/versions" \
        | jq -r '.versions[].num' 2>/dev/null \
        | grep -qx "${version}"
}

wait_for_published() {
    local pkg=$1
    local version=$2
    local attempt

    for ((attempt = 1; attempt <= WAIT_ATTEMPTS; attempt++)); do
        if is_version_published "$pkg" "$version"; then
            echo "${pkg} ${version} is indexed on crates.io."
            return 0
        fi
        if ((attempt < WAIT_ATTEMPTS)); then
            echo "Waiting for ${pkg} ${version} to enter the crates.io index (${attempt}/${WAIT_ATTEMPTS})..."
            sleep "$WAIT_INTERVAL"
        fi
    done

    echo "error: timed out waiting for ${pkg} ${version} on crates.io." >&2
    return 1
}

publish_package() {
    local pkg=$1
    local version

    version=$(package_version "$pkg")

    echo "----------------------------------------------------------------"
    echo "Checking ${pkg} ${version}..."
    echo "----------------------------------------------------------------"

    if is_version_published "$pkg" "$version"; then
        echo "${pkg} ${version} is already published, skipping."
        return 0
    fi

    echo "Publishing ${pkg} ${version}..."
    if ! (cd "$pkg" && cargo publish); then
        # An upload can occasionally report failure after the version actually
        # landed (for example an already-uploaded race). Verify before giving up.
        if is_version_published "$pkg" "$version"; then
            echo "${pkg} ${version} is already published; continuing."
        else
            return 1
        fi
    fi

    # Do not publish a dependent package until crates.io can resolve this one.
    wait_for_published "$pkg" "$version"
}

main() {
    if [[ ! -f Cargo.toml || ! -f Cargo.lock ]]; then
        echo "error: run this script from the repository root." >&2
        exit 1
    fi

    require_tool cargo
    require_tool curl
    require_tool jq

    for pkg in "${packages[@]}"; do
        publish_package "$pkg"
    done

    echo "----------------------------------------------------------------"
    echo "All packages published successfully!"
    echo "----------------------------------------------------------------"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
