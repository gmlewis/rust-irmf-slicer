#!/bin/bash
#
# This script installs all CLI tools in this workspace using cargo install.

# Exit on any error
set -e

# List of packages that contain CLI tools (have a src/main.rs)
packages=(
    "dlp-to-irmf"
    "binvox-to-irmf"
    "svx-to-irmf"
    "irmf-slicer-cli"
    "compress-irmf"
    "stl-to-irmf"
    "obj-to-irmf"
    "zip-to-irmf"
    "irmf-linter"
    "irmf-cal-cli"
)

echo "Installing all CLI tools..."

for pkg in "${packages[@]}"; do
    echo "----------------------------------------------------------------"
    echo "Installing ${pkg}..."
    echo "----------------------------------------------------------------"
    cargo install --path "${pkg}" --force
done

echo "----------------------------------------------------------------"
echo "All CLI tools installed successfully!"
echo "----------------------------------------------------------------"
