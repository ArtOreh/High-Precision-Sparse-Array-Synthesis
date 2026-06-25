from jax import config
# Enable float64 for scientific double-precision (crucial for 10^-16 accuracy)
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
import time
import jax.scipy.special as jsp
import numpy as np
import csv
from tqdm import tqdm
import mpmath

def get_interference(m, pts):
    """
    Calculates the interference intensity using the O(N) complex amplitude method.
    Mathematically equivalent to the squared magnitude of the coherent sum.
    """
    S = jnp.sum(jnp.exp(2j * jnp.pi * m * pts))
    return jnp.abs(S)**2 / (pts.shape[0]**2)

@jax.jit(static_argnums=(1,2, ))
def sll(points, k, itt = 5):
    """
    Proposed O(k*N) Optimization Method.
    Uses an analytical Newton's approach derived from the complex-domain Hessian.
    Transitions from O(N^2) geometric summation to O(N) complex amplitudes.
    """
    full_points = jnp.concatenate([jnp.array([0.0, 1.0]), points])
    
    step = 0.25 # 0.31830988618 < || < 1/3
    k_limit = int(k + (k % 1 > 0.5))

    m_base = jnp.arange(0, k_limit + step, step)

    def compute_i_max(m_start):
        curr_m = m_start

        # Newton's method: 5 iterations provide quadratic convergence to machine epsilon
        for _ in range(itt):
            phi = 2 * jnp.pi * curr_m * full_points

            exp_phi = jnp.exp(1j * phi)
            S = jnp.sum(exp_phi)
            Sx = jnp.sum(full_points * exp_phi)
            Sxx = jnp.sum(full_points**2 * exp_phi)

            # Analytical derivatives derived from the Hermitian form |S|^2
            num = jnp.imag(S * jnp.conj(Sx))
            den = jnp.real(S * jnp.conj(Sxx)) - jnp.abs(Sx)**2

            # Если sign вернул 0, принудительно заменяем его на 1.0
            sign_safe = jnp.where(den == 0.0, 1.0, jnp.sign(den))
            denom_stable = jnp.where(jnp.abs(den) < 1e-15, 1e-15 * sign_safe, den)


            # Optimization step: Moving towards the local maximum
            curr_m = curr_m + (1.0 / (2.0 * jnp.pi)) * num / denom_stable
        
        # Check boundary conditions and filter out-of-range candidates
        is_inside = (curr_m >= 1) & (curr_m <= k)
        val = get_interference(curr_m, full_points)
        return jnp.where(is_inside, val, 0.0)
    
    # Vectorized execution over all initial candidates
    all_results = jax.vmap(compute_i_max)(m_base)
    left_bound = get_interference(1, full_points)
    right_bound = get_interference(float(k), full_points)
    
    all_results_final = jnp.concatenate([all_results, jnp.array([left_bound, right_bound])])
    return jnp.max(all_results_final) 

@jax.jit(static_argnums=(1,2, ))
def find_max_grid_search(points, k, grid_size=1_000_000):
    """
    Optimized JAX Baseline: Brute-force Grid Search.
    Complexity: O(grid_size * N).
    Serves as a speed/accuracy benchmark against the proposed Newton method.
    """
    pts = jnp.concatenate([jnp.array([0.0, 1.0]), points])
    n_full = pts.shape[0]
    
    m_grid = jnp.linspace(1.0, 1.0 * k, grid_size)
    
    def scan_body(carry, p):
        s_re, s_im = carry
        phi = 2 * jnp.pi * m_grid * p
        return (s_re + jnp.cos(phi), s_im + jnp.sin(phi)), None

    init_sums = (jnp.zeros(grid_size, dtype=jnp.float64), 
                 jnp.zeros(grid_size, dtype=jnp.float64))
    
    # Memory-efficient reduction using lax.scan for GPU acceleration
    (final_re, final_im), _ = jax.lax.scan(scan_body, init_sums, pts)
    intensities = (final_re**2 + final_im**2) / (n_full**2)
    
    return jnp.max(intensities)


@jax.jit(static_argnums=(1, 2))
def find_max_adaptive_grid_search(points, k, depth=5):
    """
    Finds the global maximum Sidelobe Level (SLL) of a linear array 
    using a multi-level adaptive grid search implemented entirely within 
    an JAX loop structure.
    """
    pts = jnp.concatenate([jnp.array([0.0, 1.0], dtype=jnp.float64), points])
    n_full = pts.shape[0]
    
    # Establish grid density: minimum sampling frequency relative to spatial wavenumber k
    # Since k is defined as a static argument, level_size behaves as a compile-time constant for XLA
    level_size = int(k * 8) 
    
    # Define initial spatial frequency boundaries for the global grid sweep
    init_low = 1.0
    init_high = 1.0 * k

    # Inner scan loop computing normalized array factor intensities across the current subgrid
    def compute_intensities(m_grid):
        def scan_body(carry, p):
            s_re, s_im = carry
            phi = 2 * jnp.pi * m_grid * p
            return (s_re + jnp.cos(phi), s_im + jnp.sin(phi)), None

        # Initialize static tracking arrays for real and imaginary components
        init_sums = (jnp.zeros(level_size, dtype=jnp.float64), 
                     jnp.zeros(level_size, dtype=jnp.float64))
        
        (final_re, final_im), _ = jax.lax.scan(scan_body, init_sums, pts)
        return (final_re**2 + final_im**2) / (n_full**2)

    # Outer scan loop performing hierarchical domain zooming over depth levels
    def step_level(carry, _):
        low, high = carry
        
        # Discretize the refined domain segment between localized lower and upper bounds
        m_grid = jnp.linspace(low, high, level_size)
        intensities = compute_intensities(m_grid)
        
        # Identify the peak index within the current resolution scale
        best_idx = jnp.argmax(intensities)
        best_m = m_grid[best_idx]
        
        # Evaluate local step size to compute localized bounds for the next hierarchy level
        current_step = (high - low) / (level_size - 1)
        
        # Constrain search limits around the local maximum within the global physical boundary [1, k]
        new_low = jnp.clip(best_m - current_step, 1.0, 1.0 * k)
        new_high = jnp.clip(best_m + current_step, 1.0, 1.0 * k)
        
        # Pass updated boundaries to the subsequent hierarchy layer and record the max intensity
        return (new_low, new_high), jnp.max(intensities)

    # Execute the iterative adaptive grid narrowing sequence for fixed 'depth' iterations
    (final_low, final_high), max_history = jax.lax.scan(step_level, (init_low, init_high), None, length=depth)
    
    # Terminal element in max_history represents the converged global maximum at maximum grid resolution
    return max_history[-1]



def calculate_sll_ground(internal_points, k_val):
    """
    Computes the peak Sidelobe Level (SLL) of the thinned linear array factor 
    using a hybrid vectorized global search and arbitrary-precision local optimization.
    """

    # --- STEP 1: Fast vectorized grid search (NumPy, float64) ---
    # Construct the full array configuration including fixed boundary elements
    pts_np = np.array([0.0] + list(internal_points) + [1.0], dtype=np.float64)
    kc_np = 2.0 * np.pi * float(k_val)
    
    scan_n = int(k_val * 16)
    u_grid_np = np.linspace(1.0 / float(k_val), 1.0, scan_n, dtype=np.float64)
    
    # Evaluate array factor intensity across the grid via highly parallelized matrix operations
    # Outer product dimension mapping: (scan_n, 1) * (1, N) -> (scan_n, N)
    phases = kc_np * u_grid_np[:, np.newaxis] * pts_np[np.newaxis, :]
    vals_np = np.sum(np.cos(phases), axis=1)**2 + np.sum(np.sin(phases), axis=1)**2
    
    # Extract candidate local maxima indices, accounting for domain boundaries
    peak_indices = []
    if vals_np[0] > vals_np[1]:
        peak_indices.append(0)
        
    # Parallel identification of internal local maxima
    internal_peaks = (vals_np[1:-1] > vals_np[:-2]) & (vals_np[1:-1] > vals_np[2:])
    peak_indices.extend(np.where(internal_peaks)[0] + 1)
    
    if vals_np[-1] > vals_np[-2]:
        peak_indices.append(scan_n - 1)
        
    # Cast float64 candidate coordinates to high-precision mpmath objects
    peak_candidates = [mpmath.mpf(float(u_grid_np[idx])) for idx in peak_indices]

    # --- STEP 2: Arbitrary-precision local optimization (mpmath, 40 dps) ---
    mpmath.mp.dps = 40 # Set precision floor to mitigate numerical drift and catastrophic cancellation
    
    pts = [mpmath.mpf(0)] + [mpmath.mpf(p) for p in internal_points] + [mpmath.mpf(1)]
    kc = mpmath.mpf(2) * mpmath.pi * mpmath.mpf(k_val)
    
    def get_intensity(u):
        u = mpmath.mpf(u)
        re = sum(mpmath.cos(kc * p * u) for p in pts)
        im = sum(mpmath.sin(kc * p * u) for p in pts)
        return re**2 + im**2

    def get_derivative(u):
        u = mpmath.mpf(u)
        re = sum(mpmath.cos(kc * p * u) for p in pts)
        im = sum(mpmath.sin(kc * p * u) for p in pts)
        dre = sum(-kc * p * mpmath.sin(kc * p * u) for p in pts)
        dim = sum( kc * p * mpmath.cos(kc * p * u) for p in pts)
        return 2 * re * dre + 2 * im * dim

    # Include exact domain boundary evaluations in the initial peak set
    final_peaks = [get_intensity(mpmath.mpf(1)/k_val), get_intensity(mpmath.mpf(1))]

    # Perform rigorous root-finding exclusively on validated sidelobe peaks
    for cand in peak_candidates:
        try:
            # Newton-Raphson scheme utilizes float64 priors to achieve rapid 40-digit convergence
            precise_u = mpmath.findroot(get_derivative, cand, solver='newton')
            if mpmath.mpf(1)/k_val <= precise_u <= mpmath.mpf(1):
                final_peaks.append(get_intensity(precise_u))
            else:
                final_peaks.append(get_intensity(cand)) 
        except:
            final_peaks.append(get_intensity(cand))

    # Determine global maximum SLL with high-precision safety margins
    abs_max_sll = max(final_peaks)
    
    # Normalize with respect to the total element count squared
    n_elements = len(pts)
    result = abs_max_sll / (mpmath.mpf(n_elements)**2)
    
    # Revert to standard 30-digit precision before returning to the caller
    mpmath.mp.dps = 30
    return +result


N_values = np.geomspace(10, 10_000, num=15, dtype=int).tolist()
num_steps = len(N_values)
K = 100
filename = "benchmark_results.csv"

header = ["N", "Newton_Time", "Grid_Time", "Adaptive_Time", "Real_Time", "Newton_Error", "Grid_Error", "Adaptive_Error"]

with open(filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    counter_steps = 1
    for n_val in N_values:
        print(f"Evaluation [{counter_steps}]/[{num_steps}]: N = {n_val}")
        counter_steps += 1
        key = jax.random.PRNGKey(4242)
        points = jax.random.uniform(key, (n_val,))
        
        # --- JIT WARM-UP (Excluding compilation time from metrics) ---
        print("Warm-up in progress...")
        _ = sll(points, K).block_until_ready()
        _ = find_max_grid_search(points, K, grid_size=100_001).block_until_ready()
        _ = find_max_adaptive_grid_search(points, K).block_until_ready()
        
        # --- Newton Method Performance Execution ---
        print("Executing Newton Optimization...")
        t0 = time.perf_counter()
        res_newton = sll(points, K).block_until_ready()
        t_newton = time.perf_counter() - t0
        print(f"Newton Result: {res_newton:.16f}")

        # --- Grid Search Performance Execution ---
        print("Executing Baseline Grid Search...")
        t0 = time.perf_counter()
        res_grid = find_max_grid_search(points, K, grid_size=100_001).block_until_ready()
        t_grid = time.perf_counter() - t0
        print(f"Grid Result:   {res_grid:.16f}")
        
        # --- Adaptive grid Search Performance Execution (Independent) ---
        t0 = time.perf_counter()
        res_adaptive = find_max_adaptive_grid_search(points, K).block_until_ready()
        t_adaptive = time.perf_counter() - t0
        print(f"Adaptive grid Result: {mpmath.nstr(res_adaptive, 20)}")

        # --- Ground Truth Verification (Independent) ---
        print("Computing Ground Truth (mpmath)...")
        t0 = time.perf_counter()
        res_real = calculate_sll_ground(points.tolist(), K)
        t_real = time.perf_counter() - t0
        print(f"Ground Truth: {mpmath.nstr(res_real, 20)}")

        mp_res_newton = mpmath.mpf(float(res_newton))
        mp_res_grid = mpmath.mpf(float(res_grid))
        mp_res_adaptive = mpmath.mpf(float(res_adaptive))

        # Error Metrics Calculation
        err_newton = float(abs(res_real - mp_res_newton))
        err_grid = float(abs(res_real - mp_res_grid))
        err_adaptive = float(abs(res_real - mp_res_adaptive))
        
        # Terminal Output
        terminal_header = (f"{'N':>10} | {'Newton Time':>12} | {'Grid Time':>12} | {'Adaptive Time':>12} | "
                f"{'Real Time':>12} | {'Newton Err':>12} | {'Grid Err':>12} | {'Adaptive Err':>12}")

        print(terminal_header)
        print("-" * len(terminal_header))

        print(f"{n_val:10d} | {t_newton:11.6f}s | {t_grid:11.6f}s | {t_adaptive:11.6f}s | "
            f"{t_real:11.6f}s | {err_newton:12.2e} | {err_grid:12.2e} | {err_adaptive:12.2e}")
        
        # Exporting data to CSV for publication-quality plotting
        writer.writerow([n_val, t_newton, t_grid, t_adaptive, t_real, err_newton, err_grid, err_adaptive])
        f.flush()

print(f"\nBenchmark completed. Results saved to: {filename}")
