#!/usr/bin/env python3
"""
Diagnostic script to debug occupancy data loading issues.

This script helps identify why the visualization system is failing to load
occupancy data from your simulation runs.
"""

import numpy as np
import os
import sys
from pathlib import Path
import glob


def diagnose_directory(directory_path: str):
    """Diagnose a single directory for occupancy data loading issues."""
    print(f"\n🔍 DIAGNOSING: {directory_path}")
    print("=" * 60)
    
    dir_path = Path(directory_path)
    
    # Check if directory exists
    if not dir_path.exists():
        print(f"❌ Directory does not exist: {directory_path}")
        return False
    
    # List all files
    all_files = list(dir_path.glob("*"))
    print(f"📁 Directory contains {len(all_files)} files")
    
    # Look for different occupancy file patterns
    patterns = [
        "Occupancy_*.dat",
        "occupancy_*.dat", 
        "Occupancy_*.txt",
        "occupancy_*.txt",
        "*occupancy*",
        "*.dat"
    ]
    
    found_patterns = {}
    for pattern in patterns:
        files = list(dir_path.glob(pattern))
        if files:
            found_patterns[pattern] = files
    
    print(f"\n📊 File pattern analysis:")
    for pattern, files in found_patterns.items():
        print(f"  {pattern}: {len(files)} files")
        if files and len(files) <= 5:
            print(f"    Files: {[f.name for f in files]}")
        elif files:
            print(f"    First 3: {[f.name for f in files[:3]]}")
            print(f"    Last 3: {[f.name for f in files[-3:]]}")
    
    # Focus on Occupancy_*.dat files
    occupancy_files = list(dir_path.glob("Occupancy_*.dat"))
    
    if not occupancy_files:
        print(f"\n❌ No Occupancy_*.dat files found!")
        print(f"📋 All .dat files: {[f.name for f in dir_path.glob('*.dat')]}")
        return False
    
    print(f"\n✅ Found {len(occupancy_files)} Occupancy_*.dat files")
    
    # Analyze occupancy files
    numbered_files = []
    for f in occupancy_files:
        try:
            # Extract number from filename like "Occupancy_1000.dat"
            num_str = f.stem.split('_')[-1]
            if num_str.isdigit():
                numbered_files.append((int(num_str), f))
            else:
                print(f"  ⚠️  Non-numeric suffix: {f.name}")
        except Exception as e:
            print(f"  ❌ Could not parse filename {f.name}: {e}")
    
    if numbered_files:
        numbered_files.sort(key=lambda x: x[0])
        print(f"📈 Time steps range: {numbered_files[0][0]} to {numbered_files[-1][0]}")
        
        # Try to load the final file
        final_time, final_file = numbered_files[-1]
        print(f"\n🔍 Attempting to load final state: {final_file.name}")
        
        try:
            data = np.loadtxt(final_file)
            print(f"✅ Successfully loaded!")
            print(f"  📊 Shape: {data.shape}")
            print(f"  📈 Data type: {data.dtype}")
            print(f"  📉 Range: {np.min(data):.3f} to {np.max(data):.3f}")
            print(f"  🎯 Mean: {np.mean(data):.3f} ± {np.std(data):.3f}")
            
            # Check for problematic values
            if np.any(np.isnan(data)):
                print(f"  ⚠️  Contains NaN values: {np.sum(np.isnan(data))}")
            if np.any(np.isinf(data)):
                print(f"  ⚠️  Contains infinite values: {np.sum(np.isinf(data))}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load {final_file.name}: {e}")
            print(f"  File size: {final_file.stat().st_size} bytes")
            
            # Try to read first few lines
            try:
                with open(final_file, 'r') as f:
                    first_lines = [f.readline().strip() for _ in range(5)]
                print(f"  First few lines:")
                for i, line in enumerate(first_lines):
                    print(f"    {i+1}: {line[:80]}...")
            except Exception as read_error:
                print(f"  Could not read file as text: {read_error}")
            
            return False
    else:
        print(f"❌ No properly numbered occupancy files found!")
        return False


def test_helper_function():
    """Test the helper function import and signature."""
    print(f"\n🔧 TESTING HELPER FUNCTION")
    print("=" * 60)
    
    try:
        from postprocessing.helper import load_occupancy_data, calculate_metrics
        print("✅ Successfully imported helper functions")
        
        # Inspect function signature
        import inspect
        sig = inspect.signature(load_occupancy_data)
        print(f"📝 load_occupancy_data signature: {sig}")
        
        params = sig.parameters
        print(f"📋 Parameters:")
        for param_name, param in params.items():
            default = param.default if param.default != inspect.Parameter.empty else "no default"
            print(f"  {param_name}: {param.annotation} (default: {default})")
        
        return True
        
    except ImportError as e:
        print(f"❌ Could not import helper functions: {e}")
        print("  This means you're using fallback implementations")
        return False
    except Exception as e:
        print(f"❌ Error inspecting helper functions: {e}")
        return False


def find_sample_directories(base_path: str = "runs"):
    """Find sample directories to diagnose."""
    print(f"\n🔍 FINDING SAMPLE DIRECTORIES in {base_path}")
    print("=" * 60)
    
    if not os.path.exists(base_path):
        print(f"❌ Base path {base_path} does not exist")
        return []
    
    # Look for run directories
    run_dirs = []
    for item in Path(base_path).iterdir():
        if item.is_dir():
            # Check if it contains .cmd files (indicating it's a run directory)
            cmd_files = list(item.glob("*.cmd"))
            if cmd_files:
                run_dirs.append(item)
            else:
                # Check subdirectories
                for subitem in item.iterdir():
                    if subitem.is_dir():
                        subcmd_files = list(subitem.glob("*.cmd"))
                        if subcmd_files:
                            run_dirs.append(subitem)
    
    print(f"📁 Found {len(run_dirs)} directories with .cmd files")
    
    if run_dirs:
        print(f"📋 Sample directories:")
        for i, d in enumerate(run_dirs[:5]):
            print(f"  {i+1}. {d}")
        if len(run_dirs) > 5:
            print(f"  ... and {len(run_dirs) - 5} more")
    
    return run_dirs


def main():
    """Main diagnostic function."""
    print("🩺 OCCUPANCY DATA LOADING DIAGNOSTIC")
    print("This script helps diagnose why occupancy data loading is failing")
    print("=" * 80)
    
    # Test helper function
    helper_available = test_helper_function()
    
    # Find directories to test
    base_path = sys.argv[1] if len(sys.argv) > 1 else "runs"
    sample_dirs = find_sample_directories(base_path)
    
    if not sample_dirs:
        print(f"\n❌ No simulation directories found in {base_path}")
        print("Usage: python diagnose_loading.py [runs_directory]")
        return
    
    # Diagnose a few sample directories
    success_count = 0
    test_count = min(3, len(sample_dirs))
    
    for i in range(test_count):
        directory = sample_dirs[i]
        success = diagnose_directory(str(directory))
        if success:
            success_count += 1
    
    # Summary
    print(f"\n📊 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Helper functions available: {helper_available}")
    print(f"Directories tested: {test_count}")
    print(f"Successful loads: {success_count}")
    print(f"Success rate: {success_count/test_count*100:.1f}%")
    
    if success_count == 0:
        print(f"\n🚨 TROUBLESHOOTING SUGGESTIONS:")
        print("1. Check if your simulation actually completed and generated output files")
        print("2. Verify the occupancy file naming pattern (should be Occupancy_*.dat)")
        print("3. Check if the files contain valid numeric data")
        print("4. Consider updating the load_occupancy_data function signature")
        
    elif success_count < test_count:
        print(f"\n⚠️  PARTIAL SUCCESS:")
        print("Some directories loaded successfully, others failed.")
        print("This suggests inconsistent output or incomplete simulations.")
        
    else:
        print(f"\n✅ ALL TESTS PASSED!")
        print("Occupancy data loading should work properly.")
        print("If you're still having issues, the problem might be in the helper function.")


if __name__ == "__main__":
    main()
