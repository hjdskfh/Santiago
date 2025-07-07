// Initialize movement tracking array
void InitializeMovementTracking(long int total_time) {
    if (!TrackMovement || total_time <= 0 || SaveInterval <= 0) {
        MovingParticlesSize = 0;
        MovingParticlesCount = NULL;
        MovingParticlesSteps = NULL;
        XMovingParticlesSize = 0;
        XMovingParticlesCount = NULL;
        XMovingParticlesSteps = NULL;
        YMovingParticlesSize = 0;
        YMovingParticlesCount = NULL;
        YMovingParticlesSteps = NULL;
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
    XMovingParticlesSize = MovingParticlesSize;
    YMovingParticlesSize = MovingParticlesSize;
    
    // Allocate memory for total movement tracking
    MovingParticlesCount = (long int*)calloc(MovingParticlesSize, sizeof(long int));
    MovingParticlesSteps = (long int*)calloc(MovingParticlesSize, sizeof(long int));
    
    // Allocate memory for X movement tracking
    XMovingParticlesCount = (long int*)calloc(XMovingParticlesSize, sizeof(long int));
    XMovingParticlesSteps = (long int*)calloc(XMovingParticlesSize, sizeof(long int));
    
    // Allocate memory for Y movement tracking
    YMovingParticlesCount = (long int*)calloc(YMovingParticlesSize, sizeof(long int));
    YMovingParticlesSteps = (long int*)calloc(YMovingParticlesSize, sizeof(long int));
    
    if (MovingParticlesCount == NULL || MovingParticlesSteps == NULL ||
        XMovingParticlesCount == NULL || XMovingParticlesSteps == NULL ||
        YMovingParticlesCount == NULL || YMovingParticlesSteps == NULL) {
        fprintf(stderr, "Error: Could not allocate memory for movement tracking\n");
        exit(1);
    }
    
    fprintf(stderr, "Movement tracking enabled: every %ld steps, %ld tracking points (total, X, and Y)\n", 
            SaveInterval, MovingParticlesSize);
}

// Write movement statistics to file
void WriteMovementStats(void) {
    if (!TrackMovement || MovingParticlesCount == NULL || MovingParticlesSize == 0) {
        return; // No movement tracking was enabled
    }
    
    // Write total movement statistics
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
    
    // Cleanup X movement tracking
    if (XMovingParticlesCount != NULL) {
        free(XMovingParticlesCount);
        XMovingParticlesCount = NULL;
    }
    if (XMovingParticlesSteps != NULL) {
        free(XMovingParticlesSteps);
        XMovingParticlesSteps = NULL;
    }
    XMovingParticlesSize = 0;
    
    // Cleanup Y movement tracking
    if (YMovingParticlesCount != NULL) {
        free(YMovingParticlesCount);
        YMovingParticlesCount = NULL;
    }
    if (YMovingParticlesSteps != NULL) {
        free(YMovingParticlesSteps);
        YMovingParticlesSteps = NULL;
    }
    YMovingParticlesSize = 0;
}
