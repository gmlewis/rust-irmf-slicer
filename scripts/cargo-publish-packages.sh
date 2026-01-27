#!/bin/bash
#
# This script publishes all the packages in this workspace to crates.io
# in the correct dependency order.

# Exit on any error
set -e

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
# 13. zip-to-irmf (depends on volume-to-irmf)

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
    "zip-to-irmf"
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
    echo "Waiting 60 seconds for crates.io to update..."
    sleep 60
}

# Iterate through the packages and publish them
for pkg in "${packages[@]}"; do
    publish_package "${pkg}"
done

echo "----------------------------------------------------------------"
echo "All packages published successfully!"
echo "----------------------------------------------------------------"