#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Size of the mesh
#define Lx 40
#define Ly 20

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

// Global parameters
double Density; // Particle density, as N/(Lx*Ly). Its max value is nmax
double TumbRate; // Tumbling rate
long int NParticles; // Number of particles, computed from the density

// Order in which the particles are updated. It is changed randomly at every step
long int ParticleOrder[MaxNPart];

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

void InitialCondition(void)
{
	int i,j;
	int d;
	long int n;
	
	// Random seed
	srand48(837437);
	
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
	char filename[50];
	int i,j;
	int d;
	long int n;

	// Write the occupancy matrix
	sprintf(filename,"Occupancy_%05ld.dat",index);
	f=fopen(filename,"w");
	for(j=0;j<Ly;j++)
	{
		for(i=0;i<Lx;i++)
			fprintf(f,"%d ",Occupancy[i][j]);
		fprintf(f,"\n");
	}
	fclose(f);
	
	// Write the particle coordinates and directors
	sprintf(filename,"Config_%05ld.dat",index);
	f=fopen(filename,"w");
	for(n=0;n<NParticles;n++)
		fprintf(f,"%ld %d %d %d %d\n",n,PosX[n],PosY[n],DirectorX[n],DirectorY[n]);
	fclose(f);
}

int main(int argc, char **argv)
{
	long int TotalTime; // Total simulation time, in steps
	long int step;
	
	// Read the parameters from the command line
	sscanf(argv[1],"%lf",&Density);
	sscanf(argv[2],"%lf",&TumbRate);
	sscanf(argv[3],"%ld",&TotalTime);
	
	// Compute the number of particles
	NParticles=(int)(Density*Lx*Ly);
	fprintf(stderr,"Parameters Density (input %lf, real %lf), NParticles %ld, Tumbling rate %lf, TotalTime %ld\n",
	Density,(1.0*NParticles)/(Lx*Ly),NParticles,TumbRate,TotalTime);
		
	// Initialize the simulation
	InitialCondition();
	
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