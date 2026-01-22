#!/bin/bash
#
# This script publishes all the packages in this workspace to crates.io
# in the correct dependency order.

# Exit on any error
set -e

# The dependency order is:
# 1. irmf-slicer (core library, no internal dependencies)
# 2. irmf-include-resolver (utility library, no internal dependencies)
# 3. irmf-output-voxels (depends on irmf-slicer)
# 4. irmf-output-stl (depends on irmf-output-voxels and irmf-slicer)
# 5. irmf-slicer-cli (depends on all of the above)

packages=(
    "irmf-slicer"
    "irmf-include-resolver"
    "irmf-output-voxels"
    "irmf-output-stl"
    "irmf-slicer-cli"
)

# Function to publish a package
publish_package() {
    local pkg=$1
    echo "----------------------------------------------------------------"
    echo "Publishing ${pkg}..."
    echo "----------------------------------------------------------------"
    (cd "${pkg}" && cargo publish)
    
    # Wait a bit for crates.io to process the new version before the next package
    # that depends on it tries to publish.
    echo "Waiting 30 seconds for crates.io to update..."
    sleep 30
}

# Iterate through the packages and publish them
for pkg in "${packages[@]}"; do
    publish_package "${pkg}"
done

echo "----------------------------------------------------------------"
echo "All packages published successfully!"
echo "----------------------------------------------------------------"
