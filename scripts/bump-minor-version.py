#!/usr/bin/env python3
import os
import re
import argparse

def bump_version(version):
    parts = version.split('.')
    if len(parts) != 3:
        return version
    major, minor, patch = parts
    return f"{major}.{int(minor) + 1}.0"

def update_cargo_toml(force_version=None):
    root_cargo_path = "Cargo.toml"
    with open(root_cargo_path, 'r') as f:
        content = f.read()

    # Find workspace version
    match = re.search(r'\[workspace\.package\]\nversion = "(.*?)"', content)
    if not match:
        print("Could not find workspace version in Cargo.toml")
        return

    old_version = match.group(1)
    if force_version:
        new_version = force_version
        print(f"Forcing version from {old_version} to {new_version}")
    else:
        new_version = bump_version(old_version)
        print(f"Bumping version from {old_version} to {new_version}")

    if old_version == new_version:
        print(f"Version is already {new_version}. No changes needed.")
        return

    # Update workspace version
    new_content = re.sub(r'(\[workspace\.package\]\nversion = )".*?"', r'\1"' + new_version + '"', content)
    
    with open(root_cargo_path, 'w') as f:
        f.write(new_content)

    # Find all Cargo.toml files
    for root, dirs, files in os.walk('.'):
        if 'target' in dirs:
            dirs.remove('target')
        for file in files:
            if file == "Cargo.toml":
                file_path = os.path.join(root, file)
                update_dependencies(file_path, old_version, new_version)

def update_dependencies(file_path, old_version, new_version):
    with open(file_path, 'r') as f:
        content = f.read()

    # Update dependencies that look like: irmf-slicer = { version = "0.1.0", path = "../irmf-slicer" }
    # or: irmf-slicer = { version = "0.1.0", path = "../irmf-slicer", optional = true }
    # We use .*? for the version string to be more robust when forcing versions.
    pattern = r'(version = )".*?"(, path = "../)'
    # Use double backslash to avoid issues during file writing/interpolation
    replacement = r'\1' + '"' + new_version + '"' + r'\2'
    new_content = re.sub(pattern, replacement, content)

    if content != new_content:
        print(f"Updating dependencies in {file_path}")
        with open(file_path, 'w') as f:
            f.write(new_content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bump or force workspace version and update internal dependencies.")
    parser.add_argument("--force", help="Force a specific version instead of bumping minor version.")
    args = parser.parse_args()

    update_cargo_toml(force_version=args.force)
