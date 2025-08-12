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

## Typical Workflow

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

## Usage Examples

### Run a Batch of Simulations
```sh
python terminal.sh
```

### Postprocess and Visualize Results
```sh
python visualize_simulation.py runs/run_YYYYMMDD_HHMMSS 
```

### Plot Gamma/Lambda vs. Density and Amplitude
```sh
python plot_lam_gam.py
```

### Adding features
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
- For large batch runs, ensure you have sufficient disk space in `runs/` and `analysis/`.

## Contact
Feel free to ask any questions you have by writing me an email under `leavictoria.bauer@gmail.com`.
