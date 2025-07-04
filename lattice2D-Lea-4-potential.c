#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>

// Size of the mesh
#define Lx 40
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
int PosX[MaxNPart],PosY[MaxNPart]; // Position of each particle
int DirectorX[MaxNPart],DirectorY[MaxNPart]; // Cartesian componenents of the director for each particle
long int ParticleOrder[MaxNPart]; // Order in which particles are updated

// Global parameters
char RunName[100]; // Name of the run for output directory
char InitialFile[200]; // Path to initial occupancy file (optional)
char PotentialType[50]; // Type of potential to use
double Density; // Particle density, as N/(Lx*Ly). Its max value is nmax
double TumbRate; // Tumbling rate
long int NParticles; // Number of particles, computed from the density
long int SaveInterval; // Interval for saving intermediate steps (0 = no intermediate saves)

// Function pointer for potential calculation
double (*CalculateMoveProb)(int x, int y);

// Pre-calculated movement probability map
double MoveProbMap[Lx][Ly];
int MoveProbMapInitialized = 0;

// Basic functions first
double integrate_curve(double (*func)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.5 * (func(a) + func(b));
    for (int i = 1; i < n; ++i) {
        double x = a + i * h;
        sum += func(x);
    }
    return sum * h;
}

double uneven_sin_function(double x) {
    return sin(x) - 0.5 * sin(2 * x);
}

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


// Function to initialize the movement probability map
void InitializeMoveProbMap(void) {
    if (MoveProbMapInitialized) return;

	// Set interval [0, pi] and tolerance
    // Find the x value where the maximum occurs
    double x_max = golden_section_search(uneven_sin_function, 0.0, M_PI, 1e-6);
    double f_max = uneven_sin_function(x_max);

    for (int x = 0; x < Lx; x++) {
        for (int y = 0; y < Ly; y++) {
            double scale_x = ((double)x / Lx) * M_PI;
            MoveProbMap[x][y] = 1 - (uneven_sin_function(scale_x) / f_max);
            
			if (MoveProbMap[x][y] < 0.0) {
				MoveProbMap[x][y] = 0.0; // Ensure no negative probabilities
			} else if (MoveProbMap[x][y] > 1.0) {
				MoveProbMap[x][y] = 1.0; // Ensure no probabilities greater than 1
			}

            // Debug: print first few values with detailed info: x < 3 && 
            if (y == 2) {
                fprintf(stderr, "Calculating MoveProbMap[%d][%d] with scale_x = %.6f\n", x, y, scale_x);
                fprintf(stderr, "uneven_sin_function(scale_x) = %.6f\n", uneven_sin_function(scale_x));
				fprintf(stderr, "f_max = %.6f\n", f_max);
                fprintf(stderr, "MoveProbMap[%d][%d] = %.6f\n", x, y, MoveProbMap[x][y]);
            }
        }
    }
    MoveProbMapInitialized = 1;
    fprintf(stderr, "Movement probability map initialized.\n");
}

// Potential functions
double DefaultPotential(int x, int y) {
    // Default: always allow movement (100% probability)
    return 1.0;
}

double UnevenSinPotential(int x, int y) {
	// Use pre-calculated values from the map and do 1-value
	(void)y; // Suppress unused parameter warning
	if (!MoveProbMapInitialized) {
		InitializeMoveProbMap();
	}
	return MoveProbMap[x][y];
}

// Function to set the potential based on string name
void SetPotentialFunction(const char* potential_type) {
    fprintf(stderr, "DEBUG: Setting potential function to '%s'\n", potential_type);
    if (strcmp(potential_type, "default") == 0) {
        CalculateMoveProb = DefaultPotential;
        fprintf(stderr, "Using default movement probability (always allow movement)\n");
    } else if (strcmp(potential_type, "uneven-sin") == 0) {
        CalculateMoveProb = UnevenSinPotential;
        fprintf(stderr, "Using uneven sin movement probability\n");
    } else {
        fprintf(stderr, "Error: Unknown movement probability type '%s'\n", potential_type);
        fprintf(stderr, "Available types: default, uneven-sin\n");
        exit(1);
    }
    
    // Test the function pointer
    double test_prob = CalculateMoveProb(0, 0);
    fprintf(stderr, "DEBUG: Test call CalculateMoveProb(0,0) = %.6f\n", test_prob);
}

// Reorders each time the order of the particles
// Implementation of Fisher–Yates shuffle
void shuffle(long int *array, long int n) 
{
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
int LoadOccupancyFromFile(const char* filename)
{
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
void ReconstructParticlesFromOccupancy(void)
{
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

void InitialCondition(void)
{
	int i,j;
	int d;
	long int n;
	
	// Random seed
	srand48(837437);
	
	// Check if we should load from file
	if(strlen(InitialFile) > 0)
	{
		fprintf(stderr, "Loading initial condition from file: %s\n", InitialFile);
		
		// Clear the occupancy matrix first
		for(i = 0; i < Lx; i++)
			for(j = 0; j < Ly; j++)
				Occupancy[i][j] = 0;
				
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
			}
			
	#if (WALL==1)
		// If there is a wall, put these sites as already fully occupied. We use nmax+1 to distinguish to a dynamical site
		for(j=0;j<Ly;j++)
			Occupancy[0][j]=nmax+1;
	#endif
		
		// Loop over all the particles
		for(n=0;n<NParticles;n++)
		{
			// Find a random site that is not full
			do
			{
				i=(int)(drand48()*Lx);
				j=(int)(drand48()*Ly);
			}while(Occupancy[i][j]>=nmax);
			// Place the particle
			PosX[n]=i;
			PosY[n]=j;
			Occupancy[i][j]++;
			
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

// Perform one simulation cicle
void Iterate(long int step)
{
	long int n;
	int iini,jini,inew,jnew;
	int d;
	
	// Change the order over which the particles are updated
	shuffle(ParticleOrder,NParticles);
		
	// Run over all the particles
	// First, atempt to move it and later change director
	for(n=0;n<NParticles;n++)
	{
		iini=PosX[n];
		jini=PosY[n];
		inew=(iini+DirectorX[n]+Lx)%Lx; // New position with periodic boundary conditions
		jnew=(jini+DirectorY[n]+Ly)%Ly; // New position with periodic boundary conditions
		
		// Check if available and move the particle
		if(Occupancy[inew][jnew]<nmax)
		{
			// Calculate movement probability based on potential at new position
			double move_probability = CalculateMoveProb(inew, jnew);
			
			// Debug: Print movement probability
			if(step % 1000 == 0 && n < 5) { // Only print for first few particles every 1000 steps
				fprintf(stderr, "Step %ld, Particle %ld: pos (%d,%d) -> (%d,%d), move_prob = %.6f\n", 
				        step, n, iini, jini, inew, jnew, move_probability);
			}
			
			// Only move if random number is less than movement probability
			if(drand48() < move_probability)
			{
				PosX[n]=inew;
				PosY[n]=jnew;
				Occupancy[iini][jini]-=1;
				Occupancy[inew][jnew]+=1;
			}
		} // If not, do nothing
		else
		{
		}
			
		// Do tumble with probability TumbRate
		if(drand48()<TumbRate)
		{
			d=(int)(drand48()*NDirectors);
			DirectorX[n]=UnitX[d];
			DirectorY[n]=UnitY[d];
		}
	}
}

void WriteConfig(long int index)
{
	FILE *f;
	char filename[250];
	int i,j;
	int d;
	long int n;

	// Write the occupancy matrix
	sprintf(filename,"%s/Occupancy_%ld.dat",RunName,index);
	f=fopen(filename,"w");
	for(j=0;j<Ly;j++)
	{
		for(i=0;i<Lx;i++)
			fprintf(f,"%d ",Occupancy[i][j]);
		fprintf(f,"\n");
	}
	fclose(f);
	
	// Write the particle coordinates and directors
	sprintf(filename,"%s/Config_%ld.dat",RunName,index);
	f=fopen(filename,"w");
	for(n=0;n<NParticles;n++)
		fprintf(f,"%ld %d %d %d %d\n",n,PosX[n],PosY[n],DirectorX[n],DirectorY[n]);
	fclose(f);
}

int main(int argc, char **argv)
{
	long int TotalTime; // Total simulation time, in steps
	long int step;
	
	// Check command line arguments
	if(argc < 5 || argc > 8)
	{
		fprintf(stderr,"Usage: %s <density> <tumbling_rate> <total_time> <run_name> [initial_file|none] [move_prob_type] [save_interval]\n", argv[0]);
		fprintf(stderr,"  initial_file: path to initial occupancy matrix file, or 'none' for random init\n");
		fprintf(stderr,"  move_prob_type: movement probability type (default, uneven-sin)\n");
		fprintf(stderr,"  save_interval: save every N steps (0 = no intermediate saves, default = 0)\n");
		return 1;
	}
	
	// Read the parameters from the command line
	sscanf(argv[1],"%lf",&Density);
	sscanf(argv[2],"%lf",&TumbRate);
	sscanf(argv[3],"%ld",&TotalTime);
	strcpy(RunName, argv[4]);
	
	// Handle initial file
	if(argc >= 6) {
		if(strcmp(argv[5], "none") == 0) {
			InitialFile[0] = '\0'; // No initial file
		} else {
			strcpy(InitialFile, argv[5]); // Use provided file
		}
	} else {
		InitialFile[0] = '\0'; // Default: no initial file
	}
	
	// Handle movement probability type
	if(argc >= 7) {
		strcpy(PotentialType, argv[6]);
	} else if(argc == 6) {
		// Check if argv[5] is "none" or looks like a filename
		if(strcmp(argv[5], "none") == 0 || strstr(argv[5], ".") != NULL || strstr(argv[5], "/") != NULL) {
			strcpy(PotentialType, "default"); // It's a file (or none), so use default move prob
		} else {
			strcpy(PotentialType, argv[5]); // It's a move prob type, no initial file
			InitialFile[0] = '\0';
		}
	} else {
		strcpy(PotentialType, "default");
	}
	
	// Handle save interval
	if(argc == 8) {
		sscanf(argv[7], "%ld", &SaveInterval);
	} else {
		SaveInterval = 0; // Default: no intermediate saves
	}
	
	// Set the potential function
	SetPotentialFunction(PotentialType);
	
	// Create the output directory
	if(mkdir(RunName, 0755) != 0 && errno != EEXIST)
	{
		fprintf(stderr,"Error: Could not create directory '%s'\n", RunName);
		return 1;
	}
	
	// Compute the number of particles (may be overridden if loading from file)
	NParticles=(int)(Density*Lx*Ly);
	fprintf(stderr,"Run name: %s\n", RunName);
	fprintf(stderr,"Parameters Density (input %lf), Target NParticles %ld, Tumbling rate %lf, TotalTime %ld\n",
	Density,NParticles,TumbRate,TotalTime);
		
	// Initialize the simulation
	InitialCondition();
	
	// Print actual density after initialization
	fprintf(stderr,"Actual Parameters: NParticles %ld, Density %lf\n",
	NParticles,(1.0*NParticles)/(Lx*Ly));
	
	WriteConfig(-1);
	//Loop
	for(step=1;step<=TotalTime;step++)
	{
		if(step%100==0)
			fprintf(stderr,"Progress %ld of %ld steps (%0.2lf %%)\n",step,TotalTime,(100.*step)/TotalTime);
		Iterate(step);
		
		// Save intermediate steps if save interval is specified
		if(SaveInterval > 0 && step % SaveInterval == 0) {
			WriteConfig(step);
		}
	}
	WriteConfig(TotalTime);
	
}