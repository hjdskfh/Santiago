# 2D Lattice Simulation Analysis Pipeline

This repository contains tools for running, analyzing, and visualizing results from 2D lattice simulations with parameter sweeps and postprocessing.

## Main Components

- **C Simulation Code**: Runs the 2D lattice simulation, outputs results to structured folders in `runs`.
- **Shell Scripts**: Automate batch runs and parameter sweeps (`run_improved.sh`, `terminal_improved.sh`).
- **Python Postprocessing**:
  - `visualize.py`: Batch visualization and analysis entry point.
  - `postprocessing/manager.py`: Highest level functions for `visualize.py`.
  - `postprocessing/engine.py`: Core utilities for loading, processing, and plotting simulation data for `postprocessing/manager.py`
  - `plot_lam_gam_2D_3D.py`: 2D and 3D plotting of extracted gamma/lambda vs. rho and amplitude of g (move probability).

## Physics of the simulation

### for the lattice2D.c code
- The simulation is a lattice simulation, where this is the workflow:
    - given to the simulation as starting parameters: `density`, `tumble_rate`, `gamma`, `v0`, `amplitude`, `tumble rate`, `total time`, `type of potential used`
    - the number of particles corresponding to the `density` in a grid of length `L_x` x `L_y` get placed in either a random order or from a start configuration file (`nr particles` = `density` * `L_x` * `L_y`) while taking care that the amount of particles in one site doesn't extend the maximum occupancy `n_max`
    - every particle is assigned a random director of either `+x`, `-x`, `-y` or `+y`
    - at every time step:
        - for every particle on the board in random order: 
            - if particle is allowed to move in direction of its director (`n_max` not exceeded)
                - it moves with a `move probability`
            - else:
                - it doesn't move
        - at the end of every timestep: every particle will get a new director with the probability tumble rate
    - time steps are repeated until `total time` has passed
    - if `--denstity-averaging` used then a `Density_avg_start5000.dat` file will be created that which is the averaged density from step 5000 until the last step, where the averages are calculated from snapshots of every save interval or every 1/tumble rate if no save interval given --> if not this postprocessing is done in the visualize.py file in option 10

- the `move probability` is determined by the type of potential being used:
    - `default`: the `move probability` is given as `v0` (normally chosen as 0.5)
    - for all other potentials: it is important to note that the potentials only depend on the x value that the particle is at
    - `uneven-sin`: the `move probability` is based on only an uneven sin function specified by parameter `gamma`
        - one takes the function (sin(x) + Gamma * sin(2 * x)) and shifts it so the maximum is at 0
        - it gets rescaled so it is between `v0 - amplitude` and `v0 + amplitude`
        - because it is 2pi periodic the function gets scaled so the y values between 0 and 2pi now correspond to the y values between 0 and `L_x`. This is called `g(x)`.
        - the move probability is only based on the x position of every particle:
            `move probability = g(x)`
    - `director-uneven-sin`: 
        - one takes an uneven sin function (here: sin(x) + 0.3 * sin(2 * x + (pi / 3))) and shifts it so the maximum is at 0
        - it gets rescaled so it is between `-amplitude` and `amplitude`
        - The function gets scaled so the y values between 0 and 2pi now correspond to the y values between 0 and `L_x`. This is callled `g(x)`.
        - now one can calculate the move probability based on the x position and the director:
            if director `+x`: `move probability = v0 + g(x)`
            if director `-x`: `move probability = v0 + g(x)`
            else: `move probability = 0`
    - `director-symmetric-sin`:
        - the function taken at the beginning is sin(x)
        - the rest is same as in `director-uneven-sin`

### for the visualize.py code
- option 10: calculation of lambda and gamma
    - if `--density-averging` is used: rho will be taken as the `Density_avg_start5000.dat` file, where the averaging started at 5000.
    - if there is no `Density_avg_start*.dat` file available: rho will be calculated from all created snapshots from a startstep that can be given in option 10
    - there are two options to calculate gamma and lambda:
        - `kernel`: smooth rho with Nadaraya-Watson non-parametric kernel regression and calculate the derivatives of rho with the derivatives of Nadaraya-Watson non-parametric kernel regression
        - `diff`: smooth rho with Nadaraya-Watson non-parametric kernel regression and calculate the derivatives of rho with np.gradient
    - from this lambda and gamma are calculated, the script comes from Pablo. The only difference is that the potential U is not given directly but because g(x) = - gradU, g(x) is directly substracted from J/rho, when solving the evaluated equation at a and b

## Parameter and Flag setting in the different simulations

### Important Parameters of the simulation and where they can be set

| **Parameter**        | **Description**                                                              | **Where to Set**                | **Comment**                                                                                 |
|----------------------|------------------------------------------------------------------------------|-------------------------------|-------------------------------------------------------------------------------------------|
| `L_x`, `L_y`         | Grid size of the simulation                                                  | `lattice2D.c`                 | set to 200, 40                                                                           |
| `n_max`              | Maximum occupancy per site                                                   | `lattice2D.c`                 | set to 3                                                                                |
| `WALL`               | 1 if there should be a wall in the simulation                                | `lattice2D.c`                 | set to 0                                                                                |
| `density`            | Particle density                                                             | `run.sh`                       | Set under parameter settings (l.284)                                                       |
| `amplitude`          | Amplitude of move probability (g)                                            | `terminal.sh` or `run.sh`      | In `run.sh`: in `parse_arguments()` (l.19); set with `--amplitude` flag (default: 0.075)  |
| `v0`                 | Default move probability (0 < v0 < 1)                                        | `run.sh`                       | Value around which move probability varies. Must be between 0 and 1.                      |
| `gamma`              | Gamma parameter for uneven-sin potential                                     | `run.sh`                       | Used for the uneven-sin potential                                                          |
| `tumble_rates`       | Tumble rates for the simulation                                              | `run.sh`                       | Use only one value for all runs to calculate lambda/gamma over rho and amplitude           |
| `densities`          | Densities to be used for every run                                           | `run.sh`                       | Sets the densities for the simulation                                                      |
| `total_time`         | Total time steps for the simulation                                          | `run.sh`                       | 4 million steps take about 2 hours per setting of tumbling rate and density               |
| `start_tumble_rate`  | Start tumble rate for the simulation                                         | `run.sh`                       | Used to create the start configuration if `--start-config` setting is used                  |
| `seed`               | Seed parameter for the simulation                                            | `run.sh`                      |                                                                                           |

### Shows usage of flags for run.sh used in terminal.sh
```sh
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --output-dir DIR           Use custom output directory instead of 'runs', normally not used bc postprocessing assumes 'runs/'"
    echo "  --move-prob TYPE           Specify potential here movement probability type (default (move probability = v0 (default: 0.5)), uneven-sin, director-uneven-sin, director-symmetric-sin), normally director-uneven-sin used to get uneven density direction based potential"
    echo "  --start-config             Create start configuration and use it for all runs, otherwise random initialization for each run (default: random initialization)"
    echo "  --save-interval N          Save every N steps (default: Save every 1/tumble_rate)"
    echo "  --track-movement           Enable movement tracking at save intervals, so movement_stats.txt is created (needed for visualizing observable moving particles option 7 in visualize.py)"
    echo "  --track-flux               Enable flux tracking (raw accumulated values) (needed to calculate flux in postprocessing option 10)"
    echo "  --track-density            Enable density tracking (needed for postprocessing options 8,9,10)"
    echo "  --density-averaging [STEP] Enable density averaging after given step (default: 10000 if not specified) --> makes computation faster"
}
```
- You can also get these settings by running `python ./run.sh --help`


## Typical Workflow`L_x`

1. **Run Simulations**
   - Use the provided shell script `terminal.sh` to launch parameter sweeps or single runs. To know which flags are possible to run, run: `./run.sh --help "$@"`
   - Output is organized in `runs/` with subfolders for each parameter set. They are named like this: runs/run_YYYYMMDD_HHMMSS_[params]

2. **Postprocess Results**
   - Run `visualize.py` to analyze and visualize results through several options. The options are checked in this order and the first one that exists is run: 
        - run `python visualize.py /path-to-runs/runs/run-folder` for a specific folder to be analysed in `runs`
        - run `python visualize.py` and copy the folders you want to have analysed into the variable `manual_folders` at the very top of the programm
        - run `python visualize.py` and the variable `manual_folders` is not defined, the programm takes the latest run in `runs` to analyze
   - Then choose all options you want the script to run (explanations offered when running script). All options chosen are done for all folders chosen for analysis.
   - All options have a default, but for every option use_default can be set to `False` in `visualize.py`, then all setting can be set as an input.

3. **Extract Physical Quantities**
   - Results are saved as CSV and TXT files in the `analysis/` directory. This can be renamed for later visualization.


4. **Visualize Results**
   - important: Run option 10 for as many runs with the same tumbling rate and different densities, but only for one set_mu and for one range for the rho slices (set_rho_min and set_rho_max) and for one nr_of_slices. If you choose --density-averaging [STARTSTEP] then put in STARTSTEP for start_averaging_steps (only necessary for naming of folder). Then continue.
   - Use `plot_lam_gam_2D_3D.py` to generate 3D surfaces and 2D slices of gamma/lambda vs. density and amplitude for the density dependent analysis of option 10 in `visualize.py`. Input the `analysis/` directory that you want to analyze and choose in the main part, if you want to analyze both the lambda and gamma where the derivative was calculated through the difference (`diff`) or with the Naraday-Watson Regression (`kernel`)
   - Plots are saved to input directory.


## File/Folder Structure

- `runs/` — Simulation output folders (one per parameter set)
- `analysis/` — Postprocessing results and plots
- `visualize.py` — Main batch analysis script
- `plot_lam_gam_2D_3D.py` — 3D/2D plotting script for gamma/lambda
- `postprocessing/engine.py`, `postprocessing/manager.py`, `postprocessing/helper.py` — Core postprocessing modules
- `lattice2D.c` - C code that does the simulation
- `run.sh` - script that runs the simulations in `lattice2D.c` for different density, tumbling rate etc settings
- `terminal.sh` - script that runs `run.sh` with all the necessary flags, enables multiple runs to be started by one script
- `rest/lam_gam_est.ipynb` - code from Pablo on calculating lanbda and gamma when they are not densitydependent and when they are densitydependent
- `rest/plot_dat.py` - file to plot files in the runs like `Density_5000.dat`
- folders like `1108_analysis_mu1_slices40_1.3-2.2` - old analysis folder that got renamed, inside you can see which runs where being analysed

## Example: What a Run Looks Like

After running a simulation, you will find a folder in `runs/` like:

```
runs/run_20250812_153000_director-uneven-sin_track_flux_density_densavg5000_amplitude0.075/
├── d0.7_t0.11_time_gamma-0.5/                      # folder for one run with specific tumble rate and density
│   ├── Density_9.dat                               # snapshot of Density at step 9, can be used to make density evolution diagrams
│   ├── Density_18.dat
│   ├── Density_5000.dat
│   ├── Density_avg_start5000.dat                   # averaged density from step 5000 until the last step, where the averages are calculated from snapshots of every save interval or every 1/tumble rate if no save interval given
│   ├── movement_stats.txt
│   ├── MoveProbgradU_1.dat
│   ├── Occupancy_1.dat
│   ├── Occupancy_5000.dat
│   ├── Occupancy_400000.dat
│   ├── XAccumulatedFlux_5000.dat                    # accumulated flux at step 5000: in lambda and gamma calculation the file until the final step is taken minus the file from which the averaging starts (here 5000) to get the flux between 5000 and 400000
│   ├── XAccumulatedFlux_400000.dat
│   ├── ...
├── d0.72_t0.11_time_gamma-0.5/                      # all folders have the same contents 
│   ├── Density_9.dat
│   └── ...
├── ...
├── run_summary.log                                 # summary of all different lattice simulations in the run and which setting where used
├── *.cmd                                           # settings for the specific runs
└── ...
```

Each subfolder (e.g., `d0.7_t0.11_time_gamma-0.5/`) contains the output for a specific density and tumble rate. Files like `Density_5000.dat`, `Flux_5000.dat`, and `Movement_5000.dat` contain the raw simulation data. The top-level folder contains logs and settings for the whole run.



## Example: What an Analysis Directory Looks Like (visualize.py Option 10 & plot_lam_gam_2D_3D.py)

After running postprocessing and analysis (e.g., with option 10 in `visualize.py`), you will find a directory in `analysis/` like:

```
analysis/0808_analysis_mu1_slices40_1.3-2.7_without0.05/
├── 20250808_200703_director-uneven-sin_track_flux_density_densavg5000_amplitude0.075/
├── 20250808_221145_director-uneven-sin_track_flux_density_densavg5000_amplitude0.10/
├── 20250809_000701_director-uneven-sin_track_flux_density_densavg5000_amplitude0.125/
├── 20250809_015505_director-uneven-sin_track_flux_density_densavg5000_amplitude0.15/
├── 20250809_033821_director-uneven-sin_track_flux_density_densavg5000_amplitude0.175/
├── 20250809_051805_director-uneven-sin_track_flux_density_densavg5000_amplitude0.20/
├── gamma_vs_rho_amplitude_diff_densitydep_2D_amp0.075.png
├── gamma_vs_rho_amplitude_diff_densitydep_2D_amp0.1.png
├── gamma_vs_rho_amplitude_diff_densitydep_2D_amp0.125.png
├── gamma_vs_rho_amplitude_diff_densitydep_2D_amp0.15.png
├── gamma_vs_rho_amplitude_diff_densitydep_2D_amp0.175.png
├── gamma_vs_rho_amplitude_diff_densitydep_2D_amp0.2.png
├── gamma_vs_rho_amplitude_diff_densitydep.png
├── gamma_vs_rho_amplitude_kernel_densitydep_2D_amp0.075.png
├── gamma_vs_rho_amplitude_kernel_densitydep_2D_amp0.1.png
├── gamma_vs_rho_amplitude_kernel_densitydep_2D_amp0.125.png
├── gamma_vs_rho_amplitude_kernel_densitydep_2D_amp0.15.png
├── gamma_vs_rho_amplitude_kernel_densitydep_2D_amp0.175.png
├── gamma_vs_rho_amplitude_kernel_densitydep_2D_amp0.2.png
├── gamma_vs_rho_amplitude_kernel_densitydep.png
├── gamma_vs_rho_lines_diff_densitydep.png
├── gamma_vs_rho_lines_kernel_densitydep.png
├── lambda_vs_rho_amplitude_diff_densitydep_2D_amp0.075.png
├── lambda_vs_rho_amplitude_diff_densitydep_2D_amp0.1.png
├── lambda_vs_rho_amplitude_diff_densitydep_2D_amp0.125.png
├── lambda_vs_rho_amplitude_diff_densitydep_2D_amp0.15.png
├── lambda_vs_rho_amplitude_diff_densitydep_2D_amp0.175.png
├── lambda_vs_rho_amplitude_diff_densitydep_2D_amp0.2.png
├── lambda_vs_rho_amplitude_diff_densitydep.png
├── lambda_vs_rho_amplitude_kernel_densitydep_2D_amp0.075.png
├── lambda_vs_rho_amplitude_kernel_densitydep_2D_amp0.1.png
├── lambda_vs_rho_amplitude_kernel_densitydep_2D_amp0.125.png
├── lambda_vs_rho_amplitude_kernel_densitydep_2D_amp0.15.png
├── lambda_vs_rho_amplitude_kernel_densitydep_2D_amp0.175.png
├── lambda_vs_rho_amplitude_kernel_densitydep_2D_amp0.2.png
├── lambda_vs_rho_amplitude_kernel_densitydep.png
├── lambda_vs_rho_lines_diff_densitydep.png
├── lambda_vs_rho_lines_kernel_densitydep.png
```

Each subfolder (e.g., `20250808_200703_director-uneven-sin_track_flux_density_densavg5000_amplitude0.075/`) contains the results for a specific amplitude. The PNG files are plots of gamma and lambda vs. rho and amplitude, for both kernel and diff methods, and for different amplitudes. The naming convention reflects the analysis settings (e.g., number of slices, mu, density range, amplitude, etc.).

## Installation

### Install requirements
```sh
pip install -r requirements.txt
```

### Activate venv
```sh
source venv/bin/activate
```

## Usage Examples

### Activate venv
```sh
source venv/bin/activate
```

### Run a Batch of Simulations
```sh
python caffeinate -i ./terminal.sh 
```

### Postprocess and Visualize Results
```sh
python visualize_simulation.py runs/run_YYYYMMDD_HHMMSS 
```

### Plot Gamma/Lambda vs. Density and Amplitude
```sh
python plot_lam_gam.py
```

## Adding features
- If you need to add a parameter to `lattice2D.c` consult `HOW_TO_ADD_PARAMETERS.md`
- new analysing options can be added to `visualize.py` by:
    - to # Show options to user add your statement: e.g.
        ```python
        print("11. Print the smoothed density files")
        ```
    - in the following lines: change to appropriate range
        ```python
        mode_choice = input("\nEnter your choice (1-10, or comma-separated for multiple): ").strip()
        mode_choices = [c.strip() for c in mode_choice.split(',') if c.strip() in [str(i) for i in range(1, 11)]]
        ```
    - add the corresponding mode_choice and add the code below
        ```python
        if mode_choice == '11':
        ```

## Requirements
- Python 3.8+
- numpy, matplotlib, pandas, scipy

## Troubleshooting
- If you see warnings about missing files or folders, check that your simulation runs completed successfully and that the output structure matches expectations.
- For large batch runs without the flag `--density-averaging` in `terminal.sh` activated, ensure you have sufficient disk space in `runs/` and `analysis/`.
- If `python ./terminal.sh` is not executable, then run `chmod +x terminal.sh` to make it executable.

## Contact
Feel free to ask any questions you have by writing me an email under `leavictoria.bauer@gmail.com`.
