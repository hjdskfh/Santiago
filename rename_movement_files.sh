#!/bin/bash

# Postprocessing script for 2D lattice simulation movement files
# Renames XMovingParticles_*.dat to XAverageMoving_*.dat
# and optionally YMovingParticles_*.dat to YAverageMoving_*.dat

set -e  # Exit on any error

# Default values
DIRECTORY="."
DRY_RUN=false
VERBOSE=false
X_ONLY=false
Y_ONLY=false
CREATE_BACKUP=false

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Postprocess 2D lattice simulation movement files by renaming them.

Options:
    -d, --directory DIR     Directory to process (default: current directory)
    -n, --dry-run          Show what would be done without renaming files
    -v, --verbose          Print detailed information
    -x, --x-only           Only process X movement files
    -y, --y-only           Only process Y movement files
    -b, --backup           Create backup before renaming
    -h, --help             Show this help message

Examples:
    $0 --dry-run                          # Dry run on current directory
    $0 -d ./runs/simulation1 --backup     # Process specific directory with backup
    $0 --x-only                           # Process only X movement files
    $0 -v                                 # Verbose processing

EOF
}

# Function to create backup
create_backup() {
    local dir="$1"
    local timestamp=$(date "+%Y%m%d_%H%M%S")
    local backup_dir="${dir}/backup_${timestamp}"
    
    # Find movement files
    local files=($(find "$dir" -maxdepth 1 -name "*Moving*.dat" 2>/dev/null))
    
    if [ ${#files[@]} -eq 0 ]; then
        echo "No movement files found for backup in $dir"
        return 0
    fi
    
    mkdir -p "$backup_dir"
    
    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            cp "$file" "$backup_dir/"
            if [ "$VERBOSE" = true ]; then
                echo "  Backed up: $(basename "$file")"
            fi
        fi
    done
    
    echo "Backed up ${#files[@]} files to $backup_dir"
}

# Function to rename files
rename_files() {
    local dir="$1"
    local pattern="$2"
    local new_name="$3"
    
    # Find files matching the pattern
    local files=($(find "$dir" -maxdepth 1 -name "${pattern}_*.dat" 2>/dev/null))
    
    if [ ${#files[@]} -eq 0 ]; then
        if [ "$VERBOSE" = true ]; then
            echo "No files matching pattern '${pattern}_*.dat' found in $dir"
        fi
        return 0
    fi
    
    echo "Found ${#files[@]} files matching pattern '${pattern}_*.dat'"
    
    if [ "$DRY_RUN" = true ]; then
        echo "DRY RUN MODE - No files will be renamed"
    fi
    
    local renamed_count=0
    
    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            local basename=$(basename "$file")
            local suffix="${basename#${pattern}_}"
            local new_basename="${new_name}_${suffix}"
            local new_file="${dir}/${new_basename}"
            
            if [ "$VERBOSE" = true ] || [ "$DRY_RUN" = true ]; then
                echo "  $basename -> $new_basename"
            fi
            
            if [ "$DRY_RUN" = false ]; then
                if mv "$file" "$new_file"; then
                    ((renamed_count++))
                else
                    echo "Error renaming $file"
                fi
            else
                ((renamed_count++))
            fi
        fi
    done
    
    if [ "$DRY_RUN" = false ]; then
        echo "Successfully renamed $renamed_count files"
    else
        echo "Would rename $renamed_count files"
    fi
    
    return $renamed_count
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--directory)
            DIRECTORY="$2"
            shift 2
            ;;
        -n|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -x|--x-only)
            X_ONLY=true
            shift
            ;;
        -y|--y-only)
            Y_ONLY=true
            shift
            ;;
        -b|--backup)
            CREATE_BACKUP=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate arguments
if [ "$X_ONLY" = true ] && [ "$Y_ONLY" = true ]; then
    echo "Error: Cannot specify both --x-only and --y-only"
    exit 1
fi

# Convert to absolute path
DIRECTORY=$(cd "$DIRECTORY" && pwd)

# Check if directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: Directory does not exist: $DIRECTORY"
    exit 1
fi

echo "2D Lattice Simulation Movement File Postprocessor"
echo "=================================================="
echo "Processing directory: $DIRECTORY"

if [ "$DRY_RUN" = true ]; then
    echo "Running in DRY RUN mode"
fi

# Create backup if requested
if [ "$CREATE_BACKUP" = true ]; then
    echo ""
    create_backup "$DIRECTORY"
fi

# Initialize counters
x_files=0
y_files=0

# Process X movement files
if [ "$Y_ONLY" = false ]; then
    echo ""
    echo "Processing X movement files..."
    rename_files "$DIRECTORY" "XMovingParticles" "XAverageMoving"
    x_files=$?
fi

# Process Y movement files  
if [ "$X_ONLY" = false ]; then
    echo ""
    echo "Processing Y movement files..."
    rename_files "$DIRECTORY" "YMovingParticles" "YAverageMoving"
    y_files=$?
fi

# Summary
echo ""
echo "=================================================="
echo "SUMMARY:"
if [ "$Y_ONLY" = false ]; then
    echo "X movement files processed: $x_files"
fi
if [ "$X_ONLY" = false ]; then
    echo "Y movement files processed: $y_files"
fi

total_renamed=$((x_files + y_files))
if [ "$DRY_RUN" = true ]; then
    echo "Total files that would be renamed: $total_renamed"
else
    echo "Total files renamed: $total_renamed"
fi

echo "Done!"
