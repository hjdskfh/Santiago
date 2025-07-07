#!/usr/bin/env python3
"""
Postprocessing script for 2D lattice simulation movement files.

This script renames movement flux files from the simulation output:
- XMovingParticles_*.dat -> XAverageMoving_*.dat
- YMovingParticles_*.dat -> YAverageMoving_*.dat (optional)

The script can process files in the current directory or a specified directory,
and provides options for dry-run mode and selective processing.
"""

import os
import glob
import argparse
import shutil
from datetime import datetime
from pathlib import Path

def rename_movement_files(directory=".", file_pattern="XMovingParticles", new_name="XAverageMoving", 
                         dry_run=False, verbose=False):
    """
    Rename movement files from one pattern to another.
    
    Args:
        directory (str): Directory to search for files
        file_pattern (str): Pattern to match in filename (e.g., "XMovingParticles")
        new_name (str): New name to replace the pattern (e.g., "XAverageMoving")
        dry_run (bool): If True, only show what would be renamed without doing it
        verbose (bool): If True, print detailed information
    
    Returns:
        int: Number of files processed
    """
    # Convert to absolute path
    directory = os.path.abspath(directory)
    
    # Search pattern
    search_pattern = os.path.join(directory, f"{file_pattern}_*.dat")
    files_to_rename = glob.glob(search_pattern)
    
    if not files_to_rename:
        if verbose:
            print(f"No files matching pattern '{file_pattern}_*.dat' found in {directory}")
        return 0
    
    files_to_rename.sort()  # Sort for consistent ordering
    
    print(f"Found {len(files_to_rename)} files matching pattern '{file_pattern}_*.dat'")
    if dry_run:
        print("DRY RUN MODE - No files will be renamed")
    
    renamed_count = 0
    
    for old_file in files_to_rename:
        # Extract the suffix (everything after the pattern)
        basename = os.path.basename(old_file)
        suffix = basename.replace(f"{file_pattern}_", "")
        
        # Create new filename
        new_basename = f"{new_name}_{suffix}"
        new_file = os.path.join(directory, new_basename)
        
        if verbose or dry_run:
            print(f"  {basename} -> {new_basename}")
        
        if not dry_run:
            try:
                shutil.move(old_file, new_file)
                renamed_count += 1
            except Exception as e:
                print(f"Error renaming {old_file}: {e}")
        else:
            renamed_count += 1
    
    if not dry_run:
        print(f"Successfully renamed {renamed_count} files")
    else:
        print(f"Would rename {renamed_count} files")
    
    return renamed_count

def backup_files(directory=".", pattern="*Moving*.dat", backup_suffix=None):
    """
    Create backup copies of movement files before renaming.
    
    Args:
        directory (str): Directory containing files
        pattern (str): File pattern to backup
        backup_suffix (str): Suffix for backup files (default: timestamp)
    
    Returns:
        int: Number of files backed up
    """
    if backup_suffix is None:
        backup_suffix = datetime.now().strftime("backup_%Y%m%d_%H%M%S")
    
    directory = os.path.abspath(directory)
    search_pattern = os.path.join(directory, pattern)
    files_to_backup = glob.glob(search_pattern)
    
    if not files_to_backup:
        print(f"No files matching pattern '{pattern}' found for backup")
        return 0
    
    backup_dir = os.path.join(directory, f"backup_{backup_suffix}")
    os.makedirs(backup_dir, exist_ok=True)
    
    backed_up = 0
    for file_path in files_to_backup:
        basename = os.path.basename(file_path)
        backup_path = os.path.join(backup_dir, basename)
        try:
            shutil.copy2(file_path, backup_path)
            backed_up += 1
        except Exception as e:
            print(f"Error backing up {file_path}: {e}")
    
    print(f"Backed up {backed_up} files to {backup_dir}")
    return backed_up

def process_run_directory(run_dir, x_only=False, y_only=False, dry_run=False, 
                         verbose=False, create_backup=False):
    """
    Process a single run directory for movement file renaming.
    
    Args:
        run_dir (str): Path to the run directory
        x_only (bool): Only process X movement files
        y_only (bool): Only process Y movement files
        dry_run (bool): Dry run mode
        verbose (bool): Verbose output
        create_backup (bool): Create backup before renaming
    
    Returns:
        dict: Results summary
    """
    results = {"x_files": 0, "y_files": 0, "backup_files": 0}
    
    if not os.path.isdir(run_dir):
        print(f"Directory does not exist: {run_dir}")
        return results
    
    if verbose:
        print(f"\nProcessing directory: {run_dir}")
    
    # Create backup if requested
    if create_backup:
        results["backup_files"] = backup_files(run_dir, "*Moving*.dat")
    
    # Process X movement files
    if not y_only:
        results["x_files"] = rename_movement_files(
            directory=run_dir,
            file_pattern="XMovingParticles",
            new_name="XAverageMoving",
            dry_run=dry_run,
            verbose=verbose
        )
    
    # Process Y movement files
    if not x_only:
        results["y_files"] = rename_movement_files(
            directory=run_dir,
            file_pattern="YMovingParticles",
            new_name="YAverageMoving",
            dry_run=dry_run,
            verbose=verbose
        )
    
    return results

def find_run_directories(base_dir=".", pattern="*"):
    """
    Find directories that might contain simulation runs.
    
    Args:
        base_dir (str): Base directory to search
        pattern (str): Pattern to match directory names
    
    Returns:
        list: List of directory paths
    """
    base_dir = os.path.abspath(base_dir)
    
    # Look for directories containing .dat files
    potential_dirs = []
    
    # Check current directory
    if glob.glob(os.path.join(base_dir, "*.dat")):
        potential_dirs.append(base_dir)
    
    # Check subdirectories
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and glob.glob(os.path.join(item_path, "*.dat")):
            potential_dirs.append(item_path)
    
    return sorted(potential_dirs)

def main():
    parser = argparse.ArgumentParser(
        description="Postprocess 2D lattice simulation movement files by renaming them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run on current directory
  python postprocess_movement_files.py --dry-run
  
  # Process specific directory with backup
  python postprocess_movement_files.py --directory ./runs/simulation1 --backup
  
  # Process only X movement files
  python postprocess_movement_files.py --x-only
  
  # Auto-discover and process all run directories
  python postprocess_movement_files.py --auto-discover --verbose
        """
    )
    
    parser.add_argument("--directory", "-d", default=".",
                       help="Directory to process (default: current directory)")
    
    parser.add_argument("--auto-discover", "-a", action="store_true",
                       help="Automatically discover and process all directories with .dat files")
    
    parser.add_argument("--x-only", action="store_true",
                       help="Only process X movement files (XMovingParticles)")
    
    parser.add_argument("--y-only", action="store_true",
                       help="Only process Y movement files (YMovingParticles)")
    
    parser.add_argument("--dry-run", "-n", action="store_true",
                       help="Show what would be done without actually renaming files")
    
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Print detailed information about processing")
    
    parser.add_argument("--backup", "-b", action="store_true",
                       help="Create backup copies of files before renaming")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.x_only and args.y_only:
        print("Error: Cannot specify both --x-only and --y-only")
        return 1
    
    print("2D Lattice Simulation Movement File Postprocessor")
    print("=" * 50)
    
    if args.auto_discover:
        directories = find_run_directories(args.directory)
        print(f"Auto-discovered {len(directories)} directories with .dat files:")
        for d in directories:
            print(f"  {d}")
    else:
        directories = [args.directory]
    
    if not directories:
        print("No directories found to process")
        return 0
    
    total_results = {"x_files": 0, "y_files": 0, "backup_files": 0}
    
    for directory in directories:
        results = process_run_directory(
            run_dir=directory,
            x_only=args.x_only,
            y_only=args.y_only,
            dry_run=args.dry_run,
            verbose=args.verbose,
            create_backup=args.backup
        )
        
        for key in total_results:
            total_results[key] += results[key]
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    if args.backup and total_results["backup_files"] > 0:
        print(f"Backup files created: {total_results['backup_files']}")
    
    if not args.y_only:
        print(f"X movement files processed: {total_results['x_files']}")
    if not args.x_only:
        print(f"Y movement files processed: {total_results['y_files']}")
    
    total_renamed = total_results['x_files'] + total_results['y_files']
    if args.dry_run:
        print(f"Total files that would be renamed: {total_renamed}")
    else:
        print(f"Total files renamed: {total_renamed}")
    
    return 0

if __name__ == "__main__":
    exit(main())
