# Debug: Printing MoveProbMap Values

The simulation code now includes debug functions to print `MoveProbMap[x][y]` values to the terminal for inspection.

## Available Functions

### `PrintMoveProbMap(int start_x, int end_x, int start_y, int end_y)`
Prints a rectangular region of the movement probability map to stderr.

**Parameters:**
- `start_x`, `end_x`: X coordinate range (inclusive)
- `start_y`, `end_y`: Y coordinate range (inclusive)

**Features:**
- Automatically validates ranges (clips to lattice bounds)
- Initializes MoveProbMap if not already done
- Shows coordinates and formatted values with 4 decimal places

### `PrintMoveProbMapSample(void)`
Convenience function that prints a 10×5 sample region (coordinates [0-9][0-4]).

## How to Use

### Method 1: Automatic Printing During Initialization
Uncomment the line in the `InitializeMoveProbMap()` function:

```c
// Print a sample of the map for debugging (uncomment to see values)
PrintMoveProbMapSample();
```

This will automatically print a sample every time the map is initialized.

### Method 2: Manual Calls
Add calls anywhere in your code after the MoveProbMap is initialized:

```c
// Print a specific region
PrintMoveProbMap(10, 20, 5, 15);

// Print a sample
PrintMoveProbMapSample();

// Print entire map (for small lattices)
PrintMoveProbMap(0, Lx-1, 0, Ly-1);
```

### Method 3: Command Line Testing
You can quickly see the values by running a short simulation:

```bash
# Test uneven-sin potential
./lattice2D-Lea-4-potential --density 0.5 --tumble-rate 0.1 --total-time 10 \
    --run-name debug_test --potential uneven-sin --gamma 2.0

# Test director-based-sin potential  
./lattice2D-Lea-4-potential --density 0.5 --tumble-rate 0.1 --total-time 10 \
    --run-name debug_test --potential director-based-sin --gamma -0.5 --g 0.8
```

## Example Output

```
MoveProbMap[0-9][0-4] for potential type 'uneven-sin':
    y\x        0        1        2        3        4        5        6        7        8        9 
      0   0.5000   0.5287   0.5573   0.5857   0.6138   0.6415   0.6688   0.6955   0.7215   0.7468 
      1   0.5000   0.5287   0.5573   0.5857   0.6138   0.6415   0.6688   0.6955   0.7215   0.7468 
      2   0.5000   0.5287   0.5573   0.5857   0.6138   0.6415   0.6688   0.6955   0.7215   0.7468 
      3   0.5000   0.5287   0.5573   0.5857   0.6138   0.6415   0.6688   0.7215   0.7468 
      4   0.5000   0.5287   0.5573   0.5857   0.6138   0.6415   0.6688   0.6955   0.7215   0.7468 
```

## Understanding the Values

- **Default potential**: All values are 0.5000 (uniform probability)
- **Uneven-sin potential**: Values vary horizontally based on the sinusoidal potential
- **Director-based-sin potential**: Values vary in both X and Y based on the director field

The values represent the probability that a particle will move in the "positive" direction (right/up) vs staying in place or moving in the "negative" direction (left/down).

## Performance Note

These debug functions use `fprintf(stderr, ...)` so they won't interfere with the main output files, but they will appear in the terminal. For large lattices, printing the entire map can be slow and produce a lot of output.
