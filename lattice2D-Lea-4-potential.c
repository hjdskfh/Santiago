#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>
#include <stdbool.h>
#include <time.h>

// Size of the mesh
#define Lx 200
#define Ly 40

#define WALL 0 // Options: 0 (false) or 1 (true)

// Maximum occupancy per site
#define nmax 3

// Maximum number of particles
#define MaxNPart (nmax)*(Lx)*(Ly)

// Possible directors
#define NDirectors 4
int UnitX[4] = {1,0,-1,0};
int UnitY[4] = {0,1,0,-1};

// Global dynamical arrays
char Occupancy[Lx][Ly]; // Occupancy of each site
int CalculatedDensity[Lx]; // Density count at each x position
int XAccumulatedFlux[Lx]; // Flux of moving particles at each site
int PosX[MaxNPart],PosY[MaxNPart]; // Position of each particle
int DirectorX[MaxNPart],DirectorY[MaxNPart]; // Cartesian componenets of the director for each particle
long int ParticleOrder[MaxNPart]; // Order in which particles are updated

// Global parameters
char RunName[100]; // Name of the run for output directory
char InitialFile[200]; // Path to initial occupancy file (optional)
char PotentialType[50]; // Type of potential to use
int PotentialTypeCode = 0; // Cached potential type: 0=default, 1=uneven-sin, 2=director-uneven-sin, 3=director-symmetric-sin
double Density; // Particle density, as N/(Lx*Ly). Its max value is nmax
double TumbRate; // Tumbling rate
long int NParticles; // Number of particles, computed from the density
long int SaveInterval; // Interval for saving intermediate steps (0 = no intermediate saves)
double PotentialLower;
double PotentialUpper;
double Gamma; // Gamma parameter for uneven sin function (strength of the second harmonic)
double G; // Global parameter for director-based potential
double X_max; // Minimum x value for potential functions (used in rescaling)

long int TotalTime; // Total simulation time (needed for array sizing)
long int *MovingParticlesCount; // Array to track moving particles per timestep
long int *MovingParticlesSteps; // Array to track which timesteps were recorded
long int MovingParticlesSize; // Size of the tracking array
int TrackMovement; // Flag: 1 = track movement, 0 = don't track

// Movement probability is determined by CalculateMovementProbability, which uses PotentialTypeCode and director arguments.

// Pre-calculated movement probability map
double MoveProbMap[Lx][Ly];
int MoveProbMapInitialized = 0;

// Structure to define a potential type
typedef struct {
    const char* name;
    void (*initialize_func)(double, double);
} PotentialDefinition;

// Forward declarations for initialization functions
void InitializeSinPotentialMap(double (*func)(double), double lower_bound, double upper_bound);
void InitializeUnevenSinMap(double lower_bound, double upper_bound);
void InitializeDirectorUnevenSinMap(double lower_bound, double upper_bound);
void InitializeDirectorSymmetricSinMap(double lower_bound, double upper_bound);
void InitializeMoveProbMap(void);

// Debug functions for printing movement probability map
void PrintMoveProbMap(int start_x, int end_x, int start_y, int end_y);
void PrintMoveProbMapSample(void);

// Forward declarations for utility functions
double golden_section_search(double (*func)(double), double a, double b, double tol);
double rescaling_function(double (*func)(double), double x, double lower_bound, double upper_bound, double f_max, double f_min);
static double uneven_sin_for_search(double x);
static double shifted_uneven_sin_for_search(double x);

// Debug function to print MoveProbMap values to terminal
// Debug function to print MoveProbMap values to terminal
void PrintMoveProbMap(int start_x, int end_x, int start_y, int end_y) {
    if (!MoveProbMapInitialized) {
        InitializeMoveProbMap();
    }
    
    // Validate ranges
    if (start_x < 0) start_x = 0;
    if (end_x >= Lx) end_x = Lx - 1;
    if (start_y < 0) start_y = 0;
    if (end_y >= Ly) end_y = Ly - 1;
    
    fprintf(stderr, "\nMoveProbMap[%d-%d][%d-%d] for potential type '%s':\n", 
            start_x, end_x, start_y, end_y, PotentialType);
    fprintf(stderr, "    y\\x ");
    
    // Print column headers
    for (int x = start_x; x <= end_x; x++) {
        fprintf(stderr, "%8d ", x);
    }
    fprintf(stderr, "\n");
    
    // Print rows
    for (int y = start_y; y <= end_y; y++) {
        fprintf(stderr, "%7d ", y);
        for (int x = start_x; x <= end_x; x++) {
            fprintf(stderr, "%8.4f ", MoveProbMap[x][y]);
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
}

// Convenience function to print a sample of the map
void PrintMoveProbMapSample(void) {
    // PrintMoveProbMap(0, 9, 0, 4);  // Print first 10x5 region for easier debugging
    // Print a few values directly for NaN/Inf check
    for (int x = 0; x < 10; x++) {
        for (int y = 0; y < 5; y++) {
            if (isnan(MoveProbMap[x][y]) || isinf(MoveProbMap[x][y])) {
                fprintf(stderr, "[ERROR] MoveProbMap[%d][%d] is invalid: %f\n", x, y, MoveProbMap[x][y]);
            }
        }
    }
}

// Registry of all available potential types - ADD NEW POTENTIALS HERE
static const PotentialDefinition POTENTIAL_REGISTRY[] = {
    {"uneven-sin", InitializeUnevenSinMap},
    {"director-uneven-sin", InitializeDirectorUnevenSinMap},
    {"director-symmetric-sin", InitializeDirectorSymmetricSinMap},
    {NULL, NULL} // Terminator
};

// Golden Section Search to find maximum
double golden_section_search(double (*func)(double), double a, double b, double tol) {
    const double gr = (sqrt(5) + 1) / 2;  // Golden ratio
    double c = b - (b - a) / gr;  // Left midpoint
    double d = a + (b - a) / gr;  // Right midpoint
    
    while (fabs(b - a) > tol) {
        if (func(c) > func(d)) {
            b = d;  // Move b to d
        } else {
            a = c;  // Move a to c
        }
        c = b - (b - a) / gr;  // Left midpoint
        d = a + (b - a) / gr;  // Right midpoint
    }
    
    // Return the midpoint of the final interval (where the maximum is located)
    return (a + b) / 2;
}

// Rescale x from [f_min, f_max] to [lower_bound, upper_bound]
double rescaling_function(double (*func)(double), double x, double lower_bound, double upper_bound, double f_max, double f_min) {
    if (fabs(f_max - f_min) < 1e-12) {
        // Avoid division by zero
        return 0.5;
    }
    return lower_bound + (func(x) - f_min) / (f_max - f_min) * (upper_bound - lower_bound);
}


// Wrapper functions for golden section search (captures gamma)
static double uneven_sin_for_search(double x) {
    return sin((x + X_max)) + Gamma * sin(2 * (x + X_max));
}

double shifted_uneven_sin_for_search(double x) {
    // Shifted version: G * (sin(x) + gamma * sin(2 * x)) + 0.5
    return G * uneven_sin_for_search(x) + 0.5;
}

double symmetric_sin_for_search(double x) {
    // Symmetric version: G * sin(x + X_max) + 0.5
    return G * sin(x + X_max) + 0.5;
}

// Unified potential initialization function
void InitializeSinPotentialMap(double (*func)(double), double lower_bound, double upper_bound) {
    double x_max = golden_section_search(func, 0.0, 2 * M_PI, 1e-6);
    if (x_max < 0.0 || x_max > 2 * M_PI) {
        fprintf(stderr, "[ERROR] Golden section search returned invalid x_max: %.6f\n", x_max);
        exit(1);
    }
    fprintf(stderr, "[PRINT] Golden section search returned invalid x_max: %.6f\n", x_max);

    double f_max = func(x_max);
    // Find true minimum over [0, 2π]
    double f_min = f_max;
    for (double x = 0.0; x <= 2*M_PI; x += 0.01) {
        double val = func(x);
        if (val < f_min) f_min = val;
    }
    // Set X_max to the maximum x value where the function is defined, so the maximum is at 0
    X_max = x_max;
    // Check if f_max and f_min are valid
    if (fabs(f_max - f_min) < 1e-12) {
        fprintf(stderr, "[ERROR] f_max == f_min (%.6f), division by zero avoided. Setting all MoveProbMap to 0.5.\n", f_max);
        for (int x = 0; x < Lx; x++) {
            for (int y = 0; y < Ly; y++) {
                MoveProbMap[x][y] = 0.5;
            }
        }
    } else {
        for (int x = 0; x < Lx; x++) {
            for (int y = 0; y < Ly; y++) {
                double scale_x = ((double)x / Lx) * 2 * M_PI; 
                MoveProbMap[x][y] = rescaling_function(func, scale_x, lower_bound, upper_bound, f_max, f_min);
                // Clamp to [0,1]
                if (isnan(MoveProbMap[x][y]) || isinf(MoveProbMap[x][y])) {
                    fprintf(stderr, "[ERROR] MoveProbMap[%d][%d] is invalid: %f\n", x, y, MoveProbMap[x][y]);
                    exit(1);
                }
                if (MoveProbMap[x][y] < 0.0) MoveProbMap[x][y] = 0.0;
                if (MoveProbMap[x][y] > 1.0) MoveProbMap[x][y] = 1.0;
            }
        }
    }
    // PrintMoveProbMap(0,200,0,0);
    fprintf(stderr, "Initialized %s potential map with bounds [%.3f, %.3f]\n", PotentialType, lower_bound, upper_bound);
}

// Wrapper functions for backward compatibility and registry
void InitializeUnevenSinMap(double lower_bound, double upper_bound) {
    InitializeSinPotentialMap(uneven_sin_for_search, lower_bound, upper_bound);
}

void InitializeDirectorUnevenSinMap(double lower_bound, double upper_bound) {
    InitializeSinPotentialMap(shifted_uneven_sin_for_search, lower_bound, upper_bound); // bounds not used for this type
}

void InitializeDirectorSymmetricSinMap(double lower_bound, double upper_bound) {
    // This potential is symmetric, so we can use the same function
    InitializeSinPotentialMap(symmetric_sin_for_search, lower_bound, upper_bound);
}


// Simple function to calculate movement probability based on potential type
// Calculate movement probability for a given site and direction.
// Uses PotentialTypeCode and director arguments. Checks array bounds for safety.
double CalculateMovementProbability(const int x, const int y, const int dir_x, const int dir_y) {
    // Array bounds check
    (void) dir_x; // dir_x is not used in this function, but kept for compatibility
    if (x < 0 || x >= Lx || y < 0 || y >= Ly) {
        fprintf(stderr, "[ERROR] CalculateMovementProbability: x/y out of bounds (%d,%d)\n", x, y);
        exit(1);
    }
    double prob = 1.0;
    switch (PotentialTypeCode) {
        case 0: // "default"
            prob = 1.0;
            break;
        case 1: // "uneven-sin"
            if (!MoveProbMapInitialized) {
                InitializeMoveProbMap();
            }
            prob = MoveProbMap[x][y];
            break;
        case 2: // "director-uneven-sin"
        case 3: // "director-symmetric-sin"
            if (dir_y == 0) {
                prob = 0.5;
            } else {
                if (!MoveProbMapInitialized) {
                    InitializeMoveProbMap();
                }
                if (dir_y > 0) {
                    prob = MoveProbMap[x][y];
                } else { // dir_y < 0
                    prob = 1 - MoveProbMap[x][y];
                }
            }
            break;
        default:
            fprintf(stderr, "Error: Unknown PotentialTypeCode %d\n", PotentialTypeCode);
            exit(1);
    }
    // NaN/Inf check
    if (isnan(prob) || isinf(prob)) {
        fprintf(stderr, "[ERROR] Movement probability is NaN/Inf at (%d,%d): %f\n", x, y, prob);
        exit(1);
    }
    return prob;
}

// Initialize the movement probability map based on potential type
void InitializeMoveProbMap(void) {
    if (MoveProbMapInitialized) return;
    
    fprintf(stderr, "Initializing %s potential (gamma = %.3f)...\n", PotentialType, Gamma);
    
    // Set flag immediately to prevent recursion
    MoveProbMapInitialized = 1;
    
    // Look up potential type in registry
    for (int i = 0; POTENTIAL_REGISTRY[i].name != NULL; i++) {
        if (strcmp(PotentialType, POTENTIAL_REGISTRY[i].name) == 0) {
            POTENTIAL_REGISTRY[i].initialize_func(PotentialLower, PotentialUpper);
            fprintf(stderr, "Movement probability map initialized.\n");
            
            // Print a sample of the map for debugging (uncomment to see values)
            // PrintMoveProbMapSample();
            
            return;
        }
    }
    
    // If we get here, the potential type was not found
    MoveProbMapInitialized = 0; // Reset flag since initialization failed
    fprintf(stderr, "Error: Unknown potential type '%s'\n", PotentialType);
    fprintf(stderr, "Available types: default");
    for (int i = 0; POTENTIAL_REGISTRY[i].name != NULL; i++) {
        fprintf(stderr, ", %s", POTENTIAL_REGISTRY[i].name);
    }
    fprintf(stderr, "\n");
    exit(1);
}

// Function to validate potential type
void SetPotentialFunction(const char* potential_type) {
    // Cache potential type as integer for faster comparisons
    if (strcmp(potential_type, "default") == 0) {
        PotentialTypeCode = 0;
        return;
    } else if (strcmp(potential_type, "uneven-sin") == 0) {
        PotentialTypeCode = 1;
        return;
    } else if (strcmp(potential_type, "director-uneven-sin") == 0) {
        PotentialTypeCode = 2;
        return;
    } else if (strcmp(potential_type, "director-symmetric-sin") == 0) {
        PotentialTypeCode = 3;
        return;
    }
    // If we get here, the potential type was not found
    fprintf(stderr, "Error: Unknown movement probability type '%s'\n", potential_type);
    exit(1);
}

// Initialize movement tracking array
void InitializeMovementTracking(long int total_time) {
    if (!TrackMovement || total_time <= 0 || SaveInterval <= 0) {
        MovingParticlesSize = 0;
        MovingParticlesCount = NULL;
        MovingParticlesSteps = NULL;
        if (TrackMovement && SaveInterval <= 0) {
            fprintf(stderr, "Movement tracking disabled: SaveInterval must be > 0 for tracking\n");
        } else if (!TrackMovement) {
            fprintf(stderr, "Movement tracking disabled\n");
        }
        return;
    }
    
    // Calculate how many tracking points we'll have
    // We track: step 0, then steps at SaveInterval, 2*SaveInterval, ..., up to total_time
    MovingParticlesSize = (total_time / SaveInterval) + 1;
    
    MovingParticlesCount = (long int*)calloc(MovingParticlesSize, sizeof(long int));
    MovingParticlesSteps = (long int*)calloc(MovingParticlesSize, sizeof(long int));
    
    if (MovingParticlesCount == NULL || MovingParticlesSteps == NULL) {
        fprintf(stderr, "Error: Could not allocate memory for movement tracking\n");
        exit(1);
    }
    
    fprintf(stderr, "Movement tracking enabled: every %ld steps, %ld tracking points\n", 
            SaveInterval, MovingParticlesSize);
}

// Write movement statistics to file
void WriteMovementStats(void) {
    if (!TrackMovement || MovingParticlesCount == NULL || MovingParticlesSize == 0) {
        return; // No movement tracking was enabled
    }
    
    char filename[256];
    sprintf(filename, "%s/movement_stats.txt", RunName);
    
    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        fprintf(stderr, "Error: Could not open %s for writing\n", filename);
        return;
    }
    
    fprintf(f, "# Timestep MovingParticles\n");
    for (long int i = 0; i < MovingParticlesSize; i++) {
        if (MovingParticlesSteps[i] > 0) { // Only write recorded timesteps (step 1 and above)
            fprintf(f, "%ld %ld\n", MovingParticlesSteps[i], MovingParticlesCount[i]);
        }
    }
    
    fclose(f);
    fprintf(stderr, "Movement statistics written to %s\n", filename);
}

// Cleanup movement tracking memory
void CleanupMovementTracking(void) {
    if (MovingParticlesCount != NULL) {
        free(MovingParticlesCount);
        MovingParticlesCount = NULL;
    }
    if (MovingParticlesSteps != NULL) {
        free(MovingParticlesSteps);
        MovingParticlesSteps = NULL;
    }
    MovingParticlesSize = 0;
}

// Reorders each time the order of the particles
// Implementation of Fisher–Yates shuffle
void shuffle(long int *array, long int n) {
    long int i,j,t;
            
    if (n > 1) 
    {
        for (i = n - 1; i > 0; i--) {
            j = (long int) (drand48()*(i+1));
            t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

// Load occupancy matrix from file
int LoadOccupancyFromFile(const char* filename){
    FILE *f;
    int i, j;
    int value;
    
    f = fopen(filename, "r");
    if(f == NULL)
    {
        fprintf(stderr, "Error: Could not open initial occupancy file '%s'\n", filename);
        return 0;
    }
    
    // Read the occupancy matrix
    for(j = 0; j < Ly; j++)
    {
        for(i = 0; i < Lx; i++)
        {
            if(fscanf(f, "%d", &value) != 1)
            {
                fprintf(stderr, "Error: Could not read occupancy value at position (%d,%d)\n", i, j);
                fclose(f);
                return 0;
            }
            Occupancy[i][j] = (char)value;
        }
    }
    
    fclose(f);
    return 1; // Success
}

// Reconstruct particle positions from occupancy matrix
void ReconstructParticlesFromOccupancy(void){
    int i, j, k;
    long int n = 0;
    int d;
    
    // Count total particles and place them
    for(i = 0; i < Lx; i++)
    {
        for(j = 0; j < Ly; j++)
        {
            // For each particle at this site
            for(k = 0; k < Occupancy[i][j] && n < MaxNPart; k++)
            {
                if(n >= NParticles)
                {
                    fprintf(stderr, "Warning: More particles in file than expected (%ld)\n", NParticles);
                    return;
                }
                
                PosX[n] = i;
                PosY[n] = j;
                
                // Assign a random director
                d = (int)(drand48() * NDirectors);
                DirectorX[n] = UnitX[d];
                DirectorY[n] = UnitY[d];
                
                n++;
            }
        }
    }
    
    // Update actual number of particles found
    NParticles = n;
    fprintf(stderr, "Reconstructed %ld particles from occupancy matrix\n", NParticles);
}

void InitialCondition(long int seed){
    int i,j;
    int d;
    long int n;
    
    // Random seed
    srand48(seed);
    
    // Check if we should load from file
    if(strlen(InitialFile) > 0)
    {
        fprintf(stderr, "Loading initial condition from file: %s\n", InitialFile);
        
        // Clear the occupancy matrix first
        for(i = 0; i < Lx; i++)
            for(j = 0; j < Ly; j++)
            {
                Occupancy[i][j] = 0;	
                CalculatedDensity[i] = 0; // Initialize density array
                XAccumulatedFlux[i] = 0; // Initialize flux matrix
            }
        
        // Load occupancy from file
        if(!LoadOccupancyFromFile(InitialFile))
        {
            fprintf(stderr, "Error loading initial condition, using random initialization instead\n");
            goto random_init;
        }
        
        // Reconstruct particle positions
        ReconstructParticlesFromOccupancy();
    }
    else
    {
        random_init:
        fprintf(stderr, "Using random initial condition\n");
        
        // Clear the occupancy matrix
        for(i=0;i<Lx;i++)
            for(j=0;j<Ly;j++)
            {
                Occupancy[i][j]=0;
                CalculatedDensity[i] = 0; // Initialize density array
                XAccumulatedFlux[i]=0; // Initialize flux matrix
            }
        
    #if (WALL==1)
        // If there is a wall, put these sites as already fully occupied. We use nmax+1 to distinguish to a dynamical site
        for(j=0;j<Ly;j++)
            Occupancy[0][j]=nmax+1;
            CalculatedDensity[0] = nmax+1; // Set density for wall

    #endif
        
        // Loop over all the particles
        for(n=0;n<NParticles;n++)
        {
            // Find a random site that is not full
            int attempts = 0;
            do{
                i=(int)(drand48()*Lx);
                j=(int)(drand48()*Ly);
                attempts++;
                if (attempts > Lx*Ly*10) {
                    fprintf(stderr, "[ERROR] Could not place particle %ld after %d attempts. Density too high for nmax=%d.\n", n, attempts, nmax);
                    exit(1);
                }
            }while(Occupancy[i][j]>=nmax);
            // Place the particle
            PosX[n]=i;
            PosY[n]=j;
            Occupancy[i][j]++;
            CalculatedDensity[i]++; // Update density for this site
            // Asign a random director
            d=(int)(drand48()*NDirectors);
            DirectorX[n]=UnitX[d];
            DirectorY[n]=UnitY[d];
        }
    }
    
    // Set the initial particle order, simple Order[n]=n
    // Later it will be changed
    for(n=0;n<NParticles;n++)
        ParticleOrder[n]=n;
}

// Perform one simulation cicle - OPTIMIZED VERSION
void Iterate(long int step){
    long int n;
    long int moving_particles_this_step = 0;
    shuffle(ParticleOrder,NParticles);
    static double *rand_buffer = NULL;
    static long int rand_buffer_size = 0;
    long int needed_randoms = NParticles * 3;
    if (rand_buffer == NULL || rand_buffer_size < needed_randoms) {
        if (rand_buffer) free(rand_buffer);
        rand_buffer_size = needed_randoms * 2;
        rand_buffer = (double*)malloc(rand_buffer_size * sizeof(double));
        if (!rand_buffer) {
            fprintf(stderr, "[ERROR] Could not allocate rand_buffer\n");
            exit(1);
        }
    }
    for (long int i = 0; i < needed_randoms; i++) {
        rand_buffer[i] = drand48();
    }
    long int rand_idx = 0;
    for(n=0;n<NParticles;n++)
    {
        long int particle_id = ParticleOrder[n];
        int icurrent = PosX[particle_id];
        int jcurrent = PosY[particle_id];
        int dir_x = DirectorX[particle_id];
        int dir_y = DirectorY[particle_id];
        // Check bounds before array access
        if (icurrent < 0 || icurrent >= Lx || jcurrent < 0 || jcurrent >= Ly) {
            fprintf(stderr, "[ERROR] Particle %ld out of bounds: x=%d, y=%d\n", n, icurrent, jcurrent);
            exit(1);
        }
       
        // Try X movement first
        if (dir_x != 0) {
            int inew = (icurrent + dir_x + Lx) % Lx;
            if (Occupancy[inew][jcurrent] < nmax) {
                double prob_x = CalculateMovementProbability(inew, jcurrent, dir_x, dir_y);
                if (rand_buffer[rand_idx++] < prob_x) {
                    Occupancy[icurrent][jcurrent]--;
                    CalculatedDensity[icurrent]--; // Decrease density for old site
                    icurrent = inew;
                    Occupancy[icurrent][jcurrent]++;
                    CalculatedDensity[icurrent]++; // Increase density for new site
                    PosX[particle_id] = icurrent;
                    moving_particles_this_step++; // Increment counter for moving particles
                    XAccumulatedFlux[icurrent] += (dir_x > 0) ? 1 : -1;
                }
            } else {
                rand_idx++; // Skip the random number we would have used
            }
        }

        // Try Y movement second (independent of X movement result)
        if (dir_y != 0) {
            int jnew = (jcurrent + dir_y + Ly) % Ly;
            if (Occupancy[icurrent][jnew] < nmax) {
                double prob_y = CalculateMovementProbability(icurrent, jnew, dir_x, dir_y);
                if (rand_buffer[rand_idx++] < prob_y) {
                    Occupancy[icurrent][jcurrent]--;
                    CalculatedDensity[icurrent]--; // Decrease density for old site
                    jcurrent = jnew;
                    Occupancy[icurrent][jcurrent]++;
                    CalculatedDensity[icurrent]++; // Increase density for new site
                    PosY[particle_id] = jcurrent;
                    moving_particles_this_step++; // Increment counter for moving particles
                    XAccumulatedFlux[icurrent] += (dir_x > 0) ? 1 : -1;
                }
            } else {
                rand_idx++; // Skip the random number we would have used
            }
        }

        // Do tumble with probability TumbRate
        if(rand_buffer[rand_idx++] < TumbRate)
        {
            int d = (int)(rand_buffer[rand_idx++] * NDirectors);
            DirectorX[particle_id] = UnitX[d];
            DirectorY[particle_id] = UnitY[d];
        }
    }
    
    // Store movement count if tracking is enabled
    
    // Store movement count if tracking is enabled 
    if (TrackMovement && MovingParticlesCount != NULL && SaveInterval > 0) {
        // Record step 1, then every SaveInterval steps (SaveInterval, 2*SaveInterval, etc.)
        if (step == 1 || step % SaveInterval == 0) {
            long int track_index;
            if (step == 1) {
                track_index = 0; // First entry is step 1
            } else {
                track_index = step / SaveInterval; // Subsequent entries
            }
            
            if (track_index >= 0 && track_index < MovingParticlesSize) {
                MovingParticlesCount[track_index] = moving_particles_this_step;
                MovingParticlesSteps[track_index] = step;
            }
        }
    }
    
    // Print movement statistics periodically
    if (step % 3000 == 0) {
        fprintf(stderr, "Step %ld\n", step);
    }
    // Free rand_buffer at the end of simulation (step == TotalTime)
    if (step == TotalTime && rand_buffer) {
        free(rand_buffer);
        rand_buffer = NULL;
        rand_buffer_size = 0;
    }
}

void WriteConfig(long int index, bool track_occupancy, bool track_density, bool track_flux)
{
    FILE *f;
    char filename[250];
    int i,j;
    long int n;

    // Write the occupancy matrix
    if (track_occupancy && (index == TotalTime || index == -1 || index == 0)) {
        sprintf(filename,"%s/Occupancy_%ld.dat",RunName,index);
        f=fopen(filename,"w");
        if (!f) {
            fprintf(stderr, "[ERROR] Could not open %s for writing occupancy\n", filename);
        } else {
            for(j=0;j<Ly;j++) {
                for(i=0;i<Lx;i++)
                    fprintf(f,"%d ",Occupancy[i][j]);
                fprintf(f,"\n");
            }
            fclose(f);
        }
    }
    // Write density
    if (track_density) {
        sprintf(filename,"%s/Density_%ld.dat",RunName,index);
        f=fopen(filename,"w");
        if (!f) {
            fprintf(stderr, "[ERROR] Could not open %s for writing density\n", filename);
        } else {
            for(i=0;i<Lx;i++)
            {  
                fprintf(f,"%.3f ", (double)CalculatedDensity[i] / Ly);
            }
            fclose(f);
        }
    }
    // Write MovingParticles
    if (track_flux && (index == TotalTime || index == -1 || index == 0)) {
        sprintf(filename,"%s/XAccumulatedFlux_%ld.dat",RunName,index);
        f=fopen(filename,"w");
        if (!f) {
            fprintf(stderr, "[ERROR] Could not open %s for writing flux\n", filename);
        } else {
            for(i=0;i<Lx;i++)
            {
                fprintf(f,"%.3f ",(double)XAccumulatedFlux[i] / (Ly*TotalTime));
            }
            fprintf(f,"\n");
        
            fclose(f);
        }
    }
    // Write the particle coordinates and directors
    if (index == -1 || index == 0) {
        sprintf(filename,"%s/Config_%ld.dat",RunName,index);
        f=fopen(filename,"w");
        if (!f) {
            fprintf(stderr, "[ERROR] Could not open %s for writing config\n", filename);
        } else {
            for(n=0;n<NParticles;n++)
                fprintf(f,"%ld %d %d %d %d\n",n,PosX[n],PosY[n],DirectorX[n],DirectorY[n]);
            fclose(f);
        }
    }
}

// Structure to hold all parameters
typedef struct {
    double density;
    double tumb_rate;
    long int total_time;
    char run_name[100];
    char initial_file[200];
    char potential_type[50];
    long int save_interval;
    int track_movement;
    double gamma;
    double g;
    bool track_occupancy;
    bool track_density;
    bool track_flux;
    double potential_lower; // Lower bound for potential (if needed)
    double potential_upper; // Upper bound for potential (if needed)
    long int seed;          // Random seed 
} SimulationParams;

// Initialize default parameters
void InitializeDefaultParams(SimulationParams *params) {
    params->density = 0.5;
    params->tumb_rate = 0.1;
    params->total_time = 1000;
    strcpy(params->run_name, "default_run");
    params->initial_file[0] = '\0';
    strcpy(params->potential_type, "default");
    params->save_interval = 0;
    params->track_movement = 0;
    params->gamma = -0.5;
    params->g = 0.3;
    params->track_occupancy = true;
    params->track_density = false;
    params->track_flux = false;
    params->potential_lower = 0.0; // Default lower bound for potential 
    params->potential_upper = 1.0; // Default upper bound for potential
    params->seed = 837437; // Default random seed
}

// Show usage information
void ShowUsage(const char* program_name) {
    fprintf(stderr, "Usage: %s [OPTIONS]\n\n", program_name);
    fprintf(stderr, "Required parameters:\n");
    fprintf(stderr, "  --density DENSITY          Particle density (0-3)\n");
    fprintf(stderr, "  --tumble-rate RATE          Tumbling rate\n");
    fprintf(stderr, "  --total-time TIME           Total simulation time\n");
    fprintf(stderr, "  --run-name NAME             Output directory name\n\n");
    
    fprintf(stderr, "Optional parameters:\n");
    fprintf(stderr, "  --initial-file FILE         Initial occupancy file (default: random)\n");
    fprintf(stderr, "  --potential TYPE            Potential type: default, uneven-sin, director-uneven-sin\n");
    fprintf(stderr, "  --save-interval N           Save every N steps (default: 1/tumble_rate)\n");
    fprintf(stderr, "  --track-movement            Enable movement tracking as Observable\n");
    fprintf(stderr, "  --gamma VALUE               Gamma parameter for sin potentials (default: -0.5)\n");
    fprintf(stderr, "  --g VALUE                   G parameter for director-uneven-sin (default: 0.3)\n");
    fprintf(stderr, "  --track-occupancy           Track occupancy matrices (default: enabled)\n");
    fprintf(stderr, "  --track-density             Track density calculations\n");
    fprintf(stderr, "  --track-flux                Track movement flux matrices\n");
    fprintf(stderr, "  --seed VALUE                Random seed (default: 837437)\n");
    fprintf(stderr, "  --help                      Show this help message\n\n");
    
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  %s --density 0.5 --tumble-rate 0.1 --total-time 10000 --run-name test\n", program_name);
    fprintf(stderr, "  %s --density 0.7 --tumble-rate 0.05 --total-time 5000 --run-name exp1 --potential director-uneven-sin --gamma -0.3 --g 0.5 --track-flux\n", program_name);
}

// Parse command line arguments
int ParseArguments(int argc, char **argv, SimulationParams *params) {
    InitializeDefaultParams(params);

    bool density_set = false, tumble_set = false, time_set = false, name_set = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            ShowUsage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "--density") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --density requires a value\n");
                return -1;
            }
            params->density = atof(argv[++i]);
            density_set = true;
        }
        else if (strcmp(argv[i], "--tumble-rate") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --tumble-rate requires a value\n");
                return -1;
            }
            params->tumb_rate = atof(argv[++i]);
            tumble_set = true;
        }
        else if (strcmp(argv[i], "--total-time") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --total-time requires a value\n");
                return -1;
            }
            params->total_time = atol(argv[++i]);
            time_set = true;
        }
        else if (strcmp(argv[i], "--run-name") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --run-name requires a value\n");
                return -1;
            }
            strncpy(params->run_name, argv[++i], sizeof(params->run_name) - 1);
            params->run_name[sizeof(params->run_name) - 1] = '\0';
            name_set = true;
        }
        else if (strcmp(argv[i], "--initial-file") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --initial-file requires a value\n");
                return -1;
            }
            strncpy(params->initial_file, argv[++i], sizeof(params->initial_file) - 1);
            params->initial_file[sizeof(params->initial_file) - 1] = '\0';
        }
        else if (strcmp(argv[i], "--potential") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --potential requires a value\n");
                return -1;
            }
            strncpy(params->potential_type, argv[++i], sizeof(params->potential_type) - 1);
            params->potential_type[sizeof(params->potential_type) - 1] = '\0';
        }
        else if (strcmp(argv[i], "--save-interval") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --save-interval requires a value\n");
                return -1;
            }
            params->save_interval = atol(argv[++i]);
        }
        else if (strcmp(argv[i], "--track-movement") == 0) {
            params->track_movement = 1;
        }
        else if (strcmp(argv[i], "--gamma") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --gamma requires a value\n");
                return -1;
            }
            params->gamma = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--g") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --g requires a value\n");
                return -1;
            }
            params->g = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--potential-lower") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --potential-lower requires a value\n");
                return -1;
            }
            params->potential_lower = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--potential-upper") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --potential-upper requires a value\n");
                return -1;
            }
            params->potential_upper = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--track-occupancy") == 0) {
            params->track_occupancy = true;
        }
        else if (strcmp(argv[i], "--track-density") == 0) {
            params->track_density = true;
        }
        else if (strcmp(argv[i], "--track-flux") == 0) {
            params->track_flux = true;
        }
        else if (strcmp(argv[i], "--seed") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --seed requires a value\n");
                return -1;
            }
            params->seed = atol(argv[++i]);
        }
        // ============ ADD NEW PARAMETERS HERE ============
        // To add a new parameter:
        // 1. Add it to SimulationParams struct above
        // 2. Add the parsing logic here following this pattern:
        //    else if (strcmp(argv[i], "--new-parameter") == 0) {
        //        if (i + 1 >= argc) {
        //            fprintf(stderr, "Error: --new-parameter requires a value\n");
        //            return -1;
        //        }
        //        params->new_parameter = atof(argv[++i]);  // or atol() for integers
        //    }
        // 3. Add it to the bash script parameter section
        // That's it! The system will automatically handle the rest.
        else {
            fprintf(stderr, "Error: Unknown option '%s'\n", argv[i]);
            fprintf(stderr, "Use --help for usage information\n");
            return -1;
        }
    }
    fprintf(stderr, "Step 0.51: Finished parsing command line arguments.\n");
    
    // Check required parameters
    if (!density_set || !tumble_set || !time_set || !name_set) {
        fprintf(stderr, "Step 0.55: Missing required parameters.\n");
        fprintf(stderr, "Error: Missing required parameters.\n");
        fprintf(stderr, "Required: --density, --tumble-rate, --total-time, --run-name\n");
        fprintf(stderr, "Use --help for full usage information\n");
        return -1;
    }
    

    // Set default save interval if not specified
    if (params->save_interval == 0) {
        params->save_interval = (long int)(1.0 / params->tumb_rate);
        if (params->save_interval < 1) params->save_interval = 1;
        fprintf(stderr, "SaveInterval not specified, using 1/TumbRate = %ld\n", params->save_interval);
    }

    
    return 1; // Success
}
    
int main(int argc, char **argv)
{
    long int step;
    SimulationParams params;

    // Parse command line arguments
    int parse_result = ParseArguments(argc, argv, &params);
    if (parse_result <= 0) {
        return (parse_result == 0) ? 0 : 1; // 0 for help, 1 for error
    }

    // Copy parameters to global variables (for compatibility with existing code)
    Density = params.density;
    TumbRate = params.tumb_rate;
    TotalTime = params.total_time;
    strcpy(RunName, params.run_name);
    strcpy(InitialFile, params.initial_file);
    strcpy(PotentialType, params.potential_type);
    SaveInterval = params.save_interval;
    TrackMovement = params.track_movement;
    Gamma = params.gamma;
    G = params.g;
    PotentialLower = params.potential_lower;
    PotentialUpper = params.potential_upper;
    X_max = 0;

    // Compute the number of particles (may be overridden if loading from file)
    NParticles=(int)(Density*Lx*Ly);

    // Initialize the simulation (pass seed from params)
    InitialCondition(params.seed);

    // Set the potential function
    SetPotentialFunction(PotentialType);

    // Create the output directory
    if(mkdir(RunName, 0755) != 0 && errno != EEXIST)
    {
        fprintf(stderr,"Error: Could not create directory '%s'\n", RunName);
        return 1;
    }

    fprintf(stderr,"Run name: %s\n", RunName);
    fprintf(stderr,"Parameters Density (input %lf), Target NParticles %ld, Tumbling rate %lf, TotalTime %ld\n",
        Density,NParticles,TumbRate,TotalTime);
    fprintf(stderr,"Movement probability type: %s, Gamma: %.3f, G: %.3f, Save interval: %ld\n", 
        PotentialType, Gamma, G, SaveInterval);
    fprintf(stderr,"Movement tracking: %s%s\n", 
        TrackMovement ? "enabled" : "disabled",
        (TrackMovement && SaveInterval > 0) ? "" : (TrackMovement ? " (but SaveInterval=0, so disabled)" : ""));
    fprintf(stderr,"Output options: Occupancy=%s, Density=%s, Flux=%s\n",
        params.track_occupancy ? "enabled" : "disabled",
        params.track_density ? "enabled" : "disabled", 
        params.track_flux ? "enabled" : "disabled");

    // Initialize movement tracking
    InitializeMovementTracking(TotalTime);

    // Print actual density after initialization
    fprintf(stderr,"Actual Parameters: NParticles %ld, Density %lf\n",
        NParticles,(1.0*NParticles)/(Lx*Ly));

    WriteConfig(-1, params.track_occupancy, params.track_density, params.track_flux);

    //Loop
    for(step=1;step<=TotalTime;step++)
    {
        if(step%3000==0)
            fprintf(stderr,"Progress %ld of %ld steps (%0.2lf %%)\n",step,TotalTime,(100.*step)/TotalTime);
        Iterate(step);

        // Save intermediate steps if save interval is specified
        if(SaveInterval > 0 && step % SaveInterval == 0) {
            WriteConfig(step, params.track_occupancy, params.track_density, params.track_flux);
        }
    }
    WriteConfig(TotalTime, params.track_occupancy, params.track_density, params.track_flux);

    // Write movement statistics before cleanup
    WriteMovementStats();

    // Cleanup movement tracking if allocated
    CleanupMovementTracking();

    return 0;
}
