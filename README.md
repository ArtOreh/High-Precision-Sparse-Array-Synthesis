# High-Precision Sparse Array Synthesis

This repository provides the numerical results and optimization engine for the synthesis of equiamplitude sparse linear arrays (SLA). The framework utilizes analytical gradient descent to minimize the Peak Sidelobe Level (PSLL).

## Repository Structure

### 1. /results
Numerical data for benchmark and large-scale cases (N = 78 to 3000).
*   **`/coordinates`**: Optimized element positions in `.txt` format. Each file contains a comma-separated list of normalized coordinates $x \in [0, 1]$. These files are provided for independent verification in third-party electromagnetic simulation software and **`visualizer.py`**: A standalone script to reproduce radiation patterns and calculate PSLL values directly from the coordinate files.
*   **`/plots`**: High-resolution radiation patterns (PDF and PNG) for each reported case, demonstrating the achieved equiripple state.
* **`/saturation_data`**: Four datasets containing PSLL values and optimized element positions(boundary points 0 and 1 are excluded) for different array average element spacing ($\rho = \{0.75, 1.0, 1.5, 2.0\}$) at a fixed main lobe width ($2/\nu$), optimization zone $u \in [1/\nu, 1]$. This data illustrates the physical saturation floor of the aperture.
* **`/error_convergence`**: Raw benchmark datasets for various configurations ($N, \nu$) containing computed intensities (Proposed Newton, uniform grid, and high-precision ground truth) along with absolute errors for each independent trial. This data tracks the exact numerical performance across individual realizations to validate the convergence behavior toward the double-precision limit and `compare_grid_and_newton.py` script which calculated this table data.
* **`/runtime_benchmark`**: Computational efficiency datasets and scripts evaluating execution speed across different algorithms.
    * `benchmark_results.csv`: Raw execution times and localized errors for Newton, grid, and adaptive search methods across various configurations ($N$).
    * `Comparison_of_execution_time_JAX.py`: The core JAX-accelerated script used to profile the algorithms, measure execution times, and regenerate the CSV table.

### 2. /article_figures
Figures and diagrams used in the paper.

### 3. Root Directory
*   **`optimizer.py`**: A Python script that executes the optimization process. By default, it runs a single iteration, displays the resulting beam pattern plot, and saves the element positions to a file named `coordinates_{N}_{K}.txt` in the root directory.

### 4. /mom_verification
Full-wave electromagnetic validation layout for the patch-antenna array design in MATLAB.
*   **`coordinates_42_42.0.txt`**: Optimized normalized spatial coordinates for the $N = 42$, $L = 42\lambda$ verification case.
*   **`patch_array_mom_solver.m`**: A MATLAB Antenna Toolbox script that imports the coordinate file and models the 2D radiation pattern using the Method of Moments (MoM).


To install the required libraries, run:
```bash
pip install -r requirements.txt
```

The `optimizer.py` script is designed for full reproducibility. 
Specific settings such as initial step size, etc. are provided as comments within the source code for each specific case.
To start the synthesis: `python optimizer.py`

To start the synthesis: 
```bash
python optimizer.py
```

# **Example:** N=132, Aperture $\nu=90.5\lambda$, PSLL $\approx$ -25.8 dB.

![Example Case 132](results/plots/132_90.5.png)


## License
Distributed under the **GNU General Public License v3 (GPLv3)**.
