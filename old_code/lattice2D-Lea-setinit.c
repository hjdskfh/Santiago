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

// Function pointer for potential calculation
double (*CalculatePotential)(int x1, int y1, int x2, int y2);

// Potential functions
double DefaultPotential(int x1, int y1, int x2, int y2) {
    // No potential interaction
    return 0.0;
}

double LennardJonesPotential(int x1, int y1, int x2, int y2) {
    // Calculate distance with periodic boundary conditions
    int dx = abs(x1 - x2);
    int dy = abs(y1 - y2);
    
    // Apply periodic boundary conditions
    if (dx > Lx/2) dx = Lx - dx;
    if (dy > Ly/2) dy = Ly - dy;
    
    double r = sqrt(dx*dx + dy*dy);
    
    // Avoid division by zero
    if (r < 0.1) return 1000.0; // High repulsion at very close distances
    
    // Simplified Lennard-Jones: V(r) = 4*epsilon*((sigma/r)^12 - (sigma/r)^6)
    double sigma = 1.0;
    double epsilon = 1.0;
    double sr6 = pow(sigma/r, 6);
    return 4*epsilon*(sr6*sr6 - sr6);
}

double CoulombPotential(int x1, int y1, int x2, int y2) {
    // Calculate distance with periodic boundary conditions
    int dx = abs(x1 - x2);
    int dy = abs(y1 - y2);
    
    // Apply periodic boundary conditions
    if (dx > Lx/2) dx = Lx - dx;
    if (dy > Ly/2) dy = Ly - dy;
    
    double r = sqrt(dx*dx + dy*dy);
    
    // Avoid division by zero
    if (r < 0.1) return 1000.0;
    
    // Simplified Coulomb: V(r) = k*q1*q2/r (assuming unit charges)
    double k = 1.0; // Coulomb constant
    return k / r;
}

// Function to set the potential based on string name
void SetPotentialFunction(const char* potential_type) {
    if (strcmp(potential_type, "default") == 0) {
        CalculatePotential = DefaultPotential;
        fprintf(stderr, "Using default potential (no interactions)\n");
    } else if (strcmp(potential_type, "lennard-jones") == 0) {
        CalculatePotential = LennardJonesPotential;
        fprintf(stderr, "Using Lennard-Jones potential\n");
    } else if (strcmp(potential_type, "coulomb") == 0) {
        CalculatePotential = CoulombPotential;
        fprintf(stderr, "Using Coulomb potential\n");
    } else {
        fprintf(stderr, "Warning: Unknown potential type '%s', using default\n", potential_type);
        CalculatePotential = DefaultPotential;
    }
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
			PosX[n]=inew;
			PosY[n]=jnew;
			Occupancy[iini][jini]-=1;
			Occupancy[inew][jnew]+=1;
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
	if(argc < 5 || argc > 7)
	{
		fprintf(stderr,"Usage: %s <density> <tumbling_rate> <total_time> <run_name> [initial_file] [potential_type]\n", argv[0]);
		fprintf(stderr,"  initial_file: optional path to initial occupancy matrix file\n");
		fprintf(stderr,"  potential_type: optional potential type (default, lennard-jones, coulomb)\n");
		return 1;
	}
	
	// Read the parameters from the command line
	sscanf(argv[1],"%lf",&Density);
	sscanf(argv[2],"%lf",&TumbRate);
	sscanf(argv[3],"%ld",&TotalTime);
	strcpy(RunName, argv[4]);
	
	// Check if initial file is provided
	if(argc >= 6 && strlen(argv[5]) > 0)
		strcpy(InitialFile, argv[5]);
	else
		InitialFile[0] = '\0'; // Empty string
	
	// Check if potential type is provided
	if(argc == 7)
		strcpy(PotentialType, argv[6]);
	else
		strcpy(PotentialType, "default");
	
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
	}
	WriteConfig(1);
	
}