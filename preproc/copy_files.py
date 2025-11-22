#!/usr/bin/env python3
"""
Copy all .ply files from source directory (including subdirectories)
to target directory, handling filename conflicts
"""

import os
import shutil
from pathlib import Path

# Source and target directories
SOURCE_DIR = Path("/mnt/cluster/workspaces/lichenyan/segmentation/ply")
TARGET_DIR = Path("/mnt/cluster/workspaces/lichenyan/cXR-DNN/data/ply")

# Ensure target directory exists
TARGET_DIR.mkdir(parents=True, exist_ok=True)

# Statistics
stats = {
    'total_found': 0,
    'copied': 0,
    'skipped': 0,
    'renamed': 0
}

# Track existing filenames (for handling conflicts)
existing_files = set(os.listdir(TARGET_DIR))

def get_unique_filename(filename, subdir_name):
    """
    Generate a unique filename, adding subdirectory name prefix or counter if conflict occurs
    """
    base_name = filename
    if base_name in existing_files:
        # Option 1: Add subdirectory name prefix
        name_without_ext = Path(base_name).stem
        ext = Path(base_name).suffix
        new_name = f"{subdir_name}_{name_without_ext}{ext}"
        
        # If still conflicts, add counter
        counter = 1
        while new_name in existing_files:
            new_name = f"{subdir_name}_{name_without_ext}_{counter}{ext}"
            counter += 1
        
        existing_files.add(new_name)
        return new_name, True  # True indicates renamed
    else:
        existing_files.add(base_name)
        return base_name, False

# Recursively find all .ply files
print(f"[INFO] Scanning source directory: {SOURCE_DIR}")
print(f"[INFO] Target directory: {TARGET_DIR}\n")

for root, dirs, files in os.walk(SOURCE_DIR):
    root_path = Path(root)
    # Get path relative to source directory (for generating unique filenames)
    rel_path = root_path.relative_to(SOURCE_DIR)
    
    for file in files:
        if file.endswith('.ply'):
            stats['total_found'] += 1
            source_file = root_path / file
            
            # Handle duplicate filenames
            if str(rel_path) == '.':
                subdir_name = 'root'
            else:
                subdir_name = str(rel_path).replace('/', '_').replace('\\', '_')
            
            target_filename, was_renamed = get_unique_filename(TARGET_DIR, file, subdir_name)
            target_file = TARGET_DIR / target_filename
            
            try:
                shutil.copy2(source_file, target_file)
                stats['copied'] += 1
                if was_renamed:
                    stats['renamed'] += 1
                    print(f"[RENAME] {source_file} -> {target_filename}")
                else:
                    if stats['copied'] % 100 == 0:
                        print(f"[INFO] Copied {stats['copied']} files...")
            except Exception as e:
                stats['skipped'] += 1
                print(f"[ERROR] Failed to copy {source_file}: {e}")

# Print statistics
print("\n" + "="*60)
print("[INFO] Copy completed!")
print(f"  Total files found: {stats['total_found']}")
print(f"  Successfully copied: {stats['copied']}")
print(f"  Renamed: {stats['renamed']}")
print(f"  Skipped/Failed: {stats['skipped']}")
print("="*60)