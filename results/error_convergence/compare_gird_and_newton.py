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
def sll(points, k, itt):
    """
    Proposed O(k*N) Optimization Method.
    Uses an analytical Newton's approach derived from the complex-domain Hessian.
    Transitions from O(N^2) geometric summation to O(N) complex amplitudes.
    """
    full_points = jnp.concatenate([jnp.array([0.0, 1.0]), points])
    
    step = 0.25
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

            # if sign return 0, replace for 1.0
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


def calculate_sll_ground(internal_points, k_val):
    """
    Calculates the Side Lobe Level (SLL) of an antenna array.
    Uses fast search on an oversampled grid (NumPy) followed by peak optimization 
    via Newton's method (mpmath) running until the specified precision is reached.
    """


    # Convert points to standard numpy array for vector operations
    pts_np = np.array([0.0] + list(internal_points) + [1.0], dtype=np.float64)
    kc_np = 2.0 * np.pi * float(k_val)
    
    scan_n = int(k_val * 16)
    u_grid_np = np.linspace(1.0 / float(k_val), 1.0, scan_n, dtype=np.float64)
    
    # Compute intensity across the grid in one vectorized pass (C-level speed)
    # Matrix multiplication: (scan_n, 1) * (1, N) -> (scan_n, N)
    phases = kc_np * u_grid_np[:, np.newaxis] * pts_np[np.newaxis, :]
    vals_np = np.sum(np.cos(phases), axis=1)**2 + np.sum(np.sin(phases), axis=1)**2
    
    # Fast search for local maxima indices (including boundaries)
    peak_indices = []
    if vals_np > vals_np:
        peak_indices.append(0)
        
    # Vectorized search for internal peaks
    internal_peaks = (vals_np[1:-1] > vals_np[:-2]) & (vals_np[1:-1] > vals_np[2:])
    peak_indices.extend(np.where(internal_peaks) + 1)
    
    if vals_np[-1] > vals_np[-2]:
        peak_indices.append(scan_n - 1)
        
    # Convert approximate coordinates to high-precision mpmath objects
    peak_candidates = [mpmath.mpf(float(u_grid_np[idx])) for idx in peak_indices]

    mpmath.mp.dps = 40 # Precision safety margin against round-off errors
    
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

    # Initialize peaks array with exact boundary values
    final_peaks = [get_intensity(mpmath.mpf(1)/k_val), get_intensity(mpmath.mpf(1))]

    # Polish only points guaranteed to be lobe peaks
    for cand in peak_candidates:
        try:
            # Newton starts from float64 approximation and converges to 40 digits
            precise_u = mpmath.findroot(get_derivative, cand, solver='newton')
            if mpmath.mpf(1)/k_val <= precise_u <= mpmath.mpf(1):
                final_peaks.append(get_intensity(precise_u))
            else:
                final_peaks.append(get_intensity(cand)) 
        except:
            final_peaks.append(get_intensity(cand))

    # Secure search for global maximum SLL in mpmath
    abs_max_sll = max(final_peaks)
    
    # Normalization to N^2 elements
    n_elements = len(pts)
    result = abs_max_sll / (mpmath.mpf(n_elements)**2)
    
    # Restore target precision (30 digits) before returning
    mpmath.mp.dps = 30
    return +result



#n, k : [(10, 22), (16, 40), (20, 15), (32, 16), (32, 100)]:
num_samples = 20_000

header = ["Res_Newton", "Res_Grid", "Res_Real", "Newton_Error", "Grid_Error"]
for n_val, K in [(10, 22), (16, 40), (20, 15), (32, 16), (32, 100)]:
    for itt in [2, 3, 4, 5]:
        
        filename = f"Table_error_{n_val}_{K}_{itt}.csv"

        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            key = jax.random.PRNGKey(42 + itt)
            max_err_newton = 0.0

            pbar = tqdm(range(num_samples))
            for _ in pbar:
                
                key, subkey = jax.random.split(key)
                points = jax.random.uniform(subkey, (n_val,))
            
                # --- Newton Method Performance Execution ---
                res_newton = sll(points, K, itt).block_until_ready()

                # --- Grid Search Performance Execution ---
                res_grid = find_max_grid_search(points, K, grid_size=100_001).block_until_ready()
                
                # --- Ground Truth Verification (Independent) ---
                res_real = calculate_sll_ground(points.tolist(), K)
                
                mp_res_newton = mpmath.mpf(float(res_newton))
                mp_res_grid = mpmath.mpf(float(res_grid))

                # Error Metrics Calculation
                err_newton = float(abs(res_real - mp_res_newton))
                err_grid = float(abs(res_real - mp_res_grid))

                if err_newton > max_err_newton:
                    print()
                    print([0] + sorted(points.tolist()) + [1])
                    print()
                    max_err_newton = err_newton

                pbar.set_postfix({
                    'curr_err': f"{err_newton:.4e}", 
                    'max_err': f"{max_err_newton:.4e}"
                })

                writer.writerow([res_newton, res_grid, res_real, err_newton, err_grid])
            f.flush()
            

        print(f"\nBenchmark completed. Results saved to: {filename}")

