## How to add parameters

### Step 1: Add to C struct (line ~614 in lattice2D-Lea-4-potential.c)
```c
typedef struct {
    // ...existing parameters...
    long int seed;          // Random seed (example of how easy it is to add parameters)
    double your_new_param;  // ADD YOUR NEW PARAMETER HERE
} SimulationParams;
```

### Step 2: Add parsing logic (line ~774 in lattice2D-Lea-4-potential.c)
```c
        else if (strcmp(argv[i], "--seed") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --seed requires a value\n");
                return -1;
            }
            params->seed = atol(argv[++i]);
        }
        // ADD YOUR NEW PARAMETER HERE:
        else if (strcmp(argv[i], "--your-new-param") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --your-new-param requires a value\n");
                return -1;
            }
            params->your_new_param = atof(argv[++i]);  // use atol() for integers
        }
```

### Step 3: Add to bash script (line ~157 in run_4_potential.sh)
```bash
# *** ADD ALL YOUR SIMULATION PARAMETERS HERE ***
gamma=-0.5              # Gamma parameter for sin potentials
g=1                     # G parameter for director-based-sin potential
potential_lower=0.0     # Lower bound for potential
potential_upper=1.0     # Upper bound for potential
seed=837437             # Random seed (example of new parameter)
your_new_param=42.0     # ADD YOUR NEW PARAMETER HERE

# And add to command building (line ~110):
    cmd="$cmd --your-new-param $your_new_param"
```

## Testing

Test your new parameter:
```bash
# See all parameters
./lattice2D-Lea-4-potential --help

# Test your parameter
./lattice2D-Lea-4-potential --density 0.5 --tumble-rate 0.1 --total-time 1000 --run-name test --your-new-param 42.0
```
