#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Script to bump version numbers across the SignalWire Agents project.
"""

import re
import os
import sys
import argparse
from pathlib import Path

def parse_version(version_str):
    """Parse a version string into a tuple of integers."""
    return tuple(map(int, version_str.split('.')))

def validate_version(version_str):
    """Validate that a version string is in the correct format (X.Y.Z)."""
    pattern = r'^\d+\.\d+\.\d+$'
    if not re.match(pattern, version_str):
        raise ValueError(f"Invalid version format: {version_str}. Expected format: X.Y.Z (e.g., 1.0.0)")
    return version_str

def increment_version(version_str):
    """Increment the patch version number."""
    parts = parse_version(version_str)
    # Increment the last part (patch version)
    new_parts = parts[:-1] + (parts[-1] + 1,)
    return '.'.join(map(str, new_parts))

def update_pyproject_toml(file_path, current_version, new_version):
    """Update version in pyproject.toml"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Update version in pyproject.toml
    updated = re.sub(
        r'version\s*=\s*"' + re.escape(current_version) + '"',
        f'version = "{new_version}"',
        content
    )
    
    with open(file_path, 'w') as f:
        f.write(updated)
    
    return current_version != updated

def update_init_py(file_path, current_version, new_version):
    """Update __version__ in __init__.py"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Update __version__ in __init__.py
    updated = re.sub(
        r'__version__\s*=\s*"' + re.escape(current_version) + '"',
        f'__version__ = "{new_version}"',
        content
    )
    
    with open(file_path, 'w') as f:
        f.write(updated)
    
    return current_version != updated

def update_changelog(file_path, current_version, new_version):
    """Update version in CHANGELOG.md by adding a new entry"""
    import datetime
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    today = datetime.date.today().strftime("%Y-%m-%d")
    new_entry = f"## [{new_version}] - {today}\n\n- Version bump\n\n"
    
    # Check if there's already a section for the new version
    if f"## [{new_version}]" in content:
        print(f"Warning: Version {new_version} already exists in CHANGELOG.md")
        return False
    
    # Add new version entry after the header
    updated = re.sub(
        r'(# Changelog.*?\n\n)',
        f'\\1{new_entry}',
        content,
        flags=re.DOTALL
    )
    
    with open(file_path, 'w') as f:
        f.write(updated)
    
    return content != updated

def update_agent_server(file_path, current_version, new_version):
    """Update version in agent_server.py"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Update version in agent_server.py
    updated = re.sub(
        r'version\s*=\s*"' + re.escape(current_version) + '"',
        f'version="{new_version}"',
        content
    )
    
    with open(file_path, 'w') as f:
        f.write(updated)
    
    return content != updated

def main():
    """Main function to bump version numbers."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Bump version numbers across the SignalWire Agents project.')
    parser.add_argument('version', nargs='?', default=None,
                        help='Target version (e.g., 1.0.0). If not provided, increments patch version.')
    args = parser.parse_args()

    # Define paths
    root_dir = Path(__file__).parent
    pyproject_path = root_dir / "pyproject.toml"
    init_path = root_dir / "signalwire_agents" / "__init__.py"
    changelog_path = root_dir / "CHANGELOG.md"
    agent_server_path = root_dir / "signalwire_agents" / "agent_server.py"

    # Get current version
    current_version = None

    # Try to get version from __init__.py first
    if init_path.exists():
        with open(init_path, 'r') as f:
            content = f.read()
            match = re.search(r'__version__\s*=\s*"([^"]+)"', content)
            if match:
                current_version = match.group(1)

    # If not found, try pyproject.toml
    if current_version is None and pyproject_path.exists():
        with open(pyproject_path, 'r') as f:
            content = f.read()
            match = re.search(r'version\s*=\s*"([^"]+)"', content)
            if match:
                current_version = match.group(1)

    if current_version is None:
        print("Error: Could not find current version in any files.")
        sys.exit(1)

    # Determine new version
    if args.version:
        try:
            new_version = validate_version(args.version)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        new_version = increment_version(current_version)

    print(f"Bumping version: {current_version} -> {new_version}")
    
    # Update files
    files_updated = 0
    
    if pyproject_path.exists():
        if update_pyproject_toml(pyproject_path, current_version, new_version):
            print(f"Updated version in {pyproject_path}")
            files_updated += 1
    
    if init_path.exists():
        if update_init_py(init_path, current_version, new_version):
            print(f"Updated version in {init_path}")
            files_updated += 1
    
    if changelog_path.exists():
        if update_changelog(changelog_path, current_version, new_version):
            print(f"Updated version in {changelog_path}")
            files_updated += 1
    
    if agent_server_path.exists():
        if update_agent_server(agent_server_path, current_version, new_version):
            print(f"Updated version in {agent_server_path}")
            files_updated += 1
    
    # Check other potential version references in the codebase
    # (this is optional but helpful for larger projects)
    for pattern in ["*.py", "*.md", "*.toml", "*.json"]:
        for file_path in root_dir.glob(f"**/{pattern}"):
            # Skip the files we've already processed
            if file_path in (pyproject_path, init_path, changelog_path, agent_server_path):
                continue
                
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                try:
                    content = f.read()
                    # Look for explicit version strings like "0.1.2" but not variable references
                    if re.search(r'[^a-zA-Z0-9_"]' + re.escape(current_version) + r'[^a-zA-Z0-9_"]', content):
                        print(f"Note: Found version string in {file_path}. You may want to check this file manually.")
                except Exception as e:
                    print(f"Warning: Could not check {file_path}: {e}")
    
    print(f"\nUpdated {files_updated} files.")
    print(f"New version: {new_version}")

if __name__ == "__main__":
    main() 