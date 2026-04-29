"""
High-Precision SLL Optimization for Uniformly Excited Sparse Arrays 
via Analytical Minimax Framework

This software implements a modified gradient descent approach to minimize 
the Peak Sidelobe Level (PSLL) in large-scale linear sparse arrays. 

Key features:
- High-precision analytical evaluation of PSLL (avoids grid-based errors).
- Modified gradient descent optimized for high-dimensional manifold navigation.
- Numba-accelerated JIT compilation for scalability.
- Support for extreme-scale synthesis (up to N=10,000).

Author: [Orekhov Artem/GitHub https://github.com/ArtOreh]
Date: April 2026
Submission: IEEE Antennas and Wireless Propagation Letters
"""

import numpy as np
import time
from numba import njit, prange
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


'''------------------------ VISUALIZATION FUNCTION ------------------------'''

def plot_interference(pts, k_max, sll_max_db):
    """
    Visualizes the provided points.
    """
    points = [0] + pts.tolist() + [1]
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12,
        "axes.labelsize": 13,
        "legend.fontsize": 14,
        "axes.grid": True
    })

    # Data processing
    pts = np.sort(np.asarray(points))
    n_points = len(pts)
    m = np.linspace(0, k_max, 100_000)
    u = m / k_max
    
    # Intensity in dB
    phases = 2j * np.pi * m.reshape(-1, 1) @ pts.reshape(1, -1)
    amplitude = np.sum(np.exp(phases), axis=1)
    intensity_db = 10 * np.log10((np.abs(amplitude)**2) / (n_points**2) + 1e-12)
    indices_below_3db = np.where(intensity_db <= -3)[0]
    u_3db = u[indices_below_3db[0]]
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axvline(x=u_3db , color='g', linestyle='-',
               label=f'BW -3dB: {u_3db*2 :.5f}')
    # Peak SLL line
    ax.axhline(y=sll_max_db, color='red', linestyle='--', lw=0.7, alpha=0.7, 
               dashes=(6, 3), label=f'Peak SLL ({sll_max_db:.2f} dB)')
    
    # Intensity Pattern
    ax.plot(u, intensity_db, color='#005b96', lw=1.5, label='Intensity Pattern')
        
    # Axes limits
    ax.set_xlabel('Normalized Direction $u$', fontsize=18)
    ax.set_ylabel('Intensity [dB]', labelpad=5, fontsize=18)
    ax.set_xlim(0, 1)
    ax.set_ylim(-65, 5)

    # Major ticks every 10 dB
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    
    # Creating log-spaced minor ticks within each 10dB interval
    # These represent 10*log10([2, 3, 4, 5, 6, 7, 8, 9]) relative to each decade
    log_steps = 10 * np.log10(np.arange(2, 10)) 
    minor_ticks = []
    for level in range(-70, 10, 10):
        minor_ticks.extend(level + log_steps)
    
    ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))

    # X-axis major ticks
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    
    ax.grid(visible=True, which='major', color='#999999',
        linestyle='-', lw=0.6, alpha=0.7)
    ax.grid(visible=True, which='minor', color='#cccccc',
        linestyle='--', lw=0.4, alpha=0.5)
    
    ax.legend(loc='upper right', frameon=True, framealpha=1, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig('interference_plot.pdf', bbox_inches='tight')
    plt.show()

'''----------------------- PSLL CALCULATION FUNCTIONS -----------------------'''

@njit
def find_local_max_newton(m_start, full_coords, k_val):
    """
    Finds the exact peak location near a given point using Newton's method.
    Uses 1st and 2nd order derivatives of the intensity function.
    """
    curr_m = np.float64(m_start)
    
    # 5 iterations of Newton's method are typically sufficient for convergence
    for _ in range(5):
        phases = (2.0 * np.pi) * curr_m * full_coords
        c = np.cos(phases)
        s = np.sin(phases)
        
        # Real and Imaginary parts of the Array Factor (A)
        A_re = np.sum(c)
        A_im = np.sum(s)
        
        # First derivatives with respect to m
        w = 2.0 * np.pi * full_coords
        Ap_re = -np.sum(w * s)
        Ap_im =  np.sum(w * c)
        
        # Second derivatives with respect to m
        w2 = w**2
        App_re = -np.sum(w2 * c)
        App_im = -np.sum(w2 * s)
        
        # First derivative of Intensity I = |A|^2: dI = 2 * Re(A_conj * Ap)
        d1 = A_re * Ap_re + A_im * Ap_im
        # Second derivative of Intensity: d2 = 2 * (Re(A_conj * App) + |Ap|^2)
        d2 = (A_re * App_re + A_im * App_im) + (Ap_re**2 + Ap_im**2)
        
        if np.abs(d2) < 1e-12:
            break
            
        curr_m -= d1 / d2

    # Map scaled direction 'm' back to 'u' space (sin theta)
    # if peak less than BEAM_WIDTH_FACTOR/k and more than 1/k_val
    # It is still accounted for to prevent main lobe splitting
    u_peak = min(curr_m / k_val, 1.0)

    if u_peak < 1.0/k_val: u_peak = 1.0/k_val*BEAM_WIDTH_FACTOR

    return u_peak, compute_intensity(full_coords, u_peak * k_val)


@njit
def compute_intensity(full_coords, freq_x):
    """
    Calculates the normalized power intensity at a given spatial frequency.
    """
    phases = (2.0 * np.pi * freq_x) * full_coords
    re = np.sum(np.cos(phases))
    im = np.sum(np.sin(phases))
    return (re**2 + im**2) / (len(full_coords)**2)


@njit(parallel=True)
def calculate_peak_sll(points, k_val):
    """
    Scans the visible region to find the Global Peak Sidelobe Level (SLL).
    Uses parallel processing to evaluate different scan regions.
    """
    # Prepare coordinates including fixed boundaries at 0.0 and 1.0
    full_coords = np.zeros(len(points) + 2)
    full_coords[0] = 0.0
    full_coords[1:-1] = points
    full_coords[-1] = 1.0

    # +1 if k_val%1>0.5
    limit = int(k_val) + 1 + int(k_val%1>0.5) 
    thread_vals = np.full(limit+1, -1e12)
    thread_peaks = np.zeros(limit+1)
    
    # Multi-point check offsets to ensure no sidelobe is missed
    offsets = np.array([0.0, 0.5, 0.25, 0.75])
    

    for m in prange(limit):
        local_max_v = -1e12
        local_max_u = 1.0
        
        for offset in offsets:
            u_p, v = find_local_max_newton(float(m) + offset, full_coords, k_val)
            if v > local_max_v:
                local_max_v = v
                local_max_u = u_p
        
        thread_vals[m] = local_max_v
        thread_peaks[m] = local_max_u
    # When the beam expansion exceeds 1/k_val, a small peak may appear 
    # near the boundary this line prevents that.
    thread_peaks[-1], thread_vals[-1] = find_local_max_newton(
        BEAM_WIDTH_FACTOR + 0.125,
        full_coords,
        k_val
    )

    # Global maximum among all scanned regions
    idx = np.argmax(thread_vals)
    max_val = thread_vals[idx]
    u_peak = thread_peaks[idx]
    
    # Boundary checks (edges of the visible region)
    res_start = compute_intensity(full_coords, 1.0*BEAM_WIDTH_FACTOR)
    if res_start > max_val:
        max_val = res_start
        u_peak = 1.0/k_val*BEAM_WIDTH_FACTOR
        
    res_end = compute_intensity(full_coords, k_val)
    if res_end > max_val:
        max_val = res_end
        u_peak = 1.0
    
    return max_val, u_peak

'''----------------------- GRADIENT DESCENT FUNCTIONS -----------------------'''

@njit
def get_taylor_density(n_internal):
    """
    Generates n_internal points in the (0, 1) range
    with increased density at the center (0.5).
    """
    # For small n_internal, the Taylor distribution
    # is inferior to the uniform distribution.
    if n_internal < 50:
        return np.random.rand(n_internal)
    
    # Simulating the Taylor distribution using this approximation.
    u = np.linspace(-0.999, 0.999, n_internal)
    coords = 0.5 + 0.5 * np.sin(u * (np.pi / 2))
    # Adding noise for accelerating convergence.
    noise = 1/(n_internal + 2)
    coords += noise * np.random.rand(n_internal)
    return coords

@njit
def compute_analytical_gradient(pts, k_val, u_peak):
    """
    Computes the partial derivatives of the intensity with respect to
    each internal element position at the current peak direction u_peak.
    """
    n = len(pts)
    grad = np.zeros(n)
    kc = 2.0 * np.pi * k_val
    
    # Total AF calculation for Re/Im reference
    full_n = n + 2
    re = 1.0 + np.cos(kc * u_peak) # Contribution from fixed points at 0 and 1
    im = 0.0 + np.sin(kc * u_peak)
    
    for p in pts:
        phase = kc * p * u_peak
        re += np.cos(phase)
        im += np.sin(phase)
        
    # Element-wise gradient
    for i in range(n):
        p_i = pts[i]
        phase = kc * p_i * u_peak
        # Partial derivatives w.r.t p_i
        dre = -kc * u_peak * np.sin(phase)
        dim =  kc * u_peak * np.cos(phase)
        # Gradient of I = (Re^2 + Im^2) / N^2
        grad[i] = 2.0 * (re * dre + im * dim) / (full_n**2)
        
    return grad

@njit
def enforce_min_element_spacing(pts, k_val):
    """
    Projection operator: ensures a minimum physical gap (MIN_DIST)
    between all elements to avoid mutual coupling and maintain realizability.
    """
    d_min = MIN_DIST / k_val 
    n = len(pts)
    
    # Forward pass: push elements to the right
    for i in range(n):
        if i == 0:
            if pts[i] < d_min: pts[i] = d_min
        else:
            if pts[i] < pts[i-1] + d_min:
                pts[i] = pts[i-1] + d_min
                
    # Backward pass: push elements to the left if they exceed the aperture
    if pts[-1] > 1.0 - d_min:
        pts[-1] = 1.0 - d_min
        for i in range(n - 2, -1, -1):
            if pts[i] > pts[i+1] - d_min:
                pts[i] = pts[i+1] - d_min
    return pts


@njit
def optimize_single_run(n_internal, k_val, iterations, lr):
    """
    Single optimization trial starting from a random distribution.
    Uses Gradient Descent.
    """
    # Initial random distribution
    pts = get_taylor_density(n_internal)

    pts = enforce_min_element_spacing(pts, k_val)
    best_pts = pts.copy()
    min_sll = 1e10
    for i in range(iterations):
        sll_val, u_peak = calculate_peak_sll(pts, k_val*min(2*(i+1)/iterations, 1))
        
        if sll_val < min_sll:
            min_sll = sll_val
            best_pts = pts.copy()
            
        grad = compute_analytical_gradient(pts, k_val, u_peak)
        
        # Gradient Normalization for stable descent
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-12:
            grad = grad / gnorm * min(max(1/gnorm, MIN_JUMP), 1)

        # Descent step
        pts = pts - lr * grad
        pts = enforce_min_element_spacing(pts, k_val)
    return min_sll, best_pts


@njit
def cold_down_stage(start_pts, k_val, iterations, lr):
    current_pts = enforce_min_element_spacing(start_pts, k_val)
    min_sll = 1e10
    best_pts = current_pts.copy()
    for i in range(iterations):
        sll_val, u_peak = calculate_peak_sll(current_pts, k_val)
        
        if sll_val < min_sll:
            min_sll = sll_val
            best_pts = current_pts.copy()
        
        grad = compute_analytical_gradient(current_pts, k_val, u_peak)
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-12:
            grad = grad / gnorm * min(max(1/gnorm, lr), 1)

        current_pts = current_pts - lr * grad
        lr *= 0.99

        current_pts = enforce_min_element_spacing(current_pts, k_val)
            
    return min_sll, best_pts

'''---------------------- CONFIGURATIONS FROM ARTICLE ----------------------'''

#####################
# N = 1000, K = 750.0
# iterations_main = 100_000
# iterations_refinement = 5_000
# learning_rate_main = 1e-3
#####################
# N = 600, K = 450.0
# iterations_main = 70_000
# iterations_refinement = 5_000
# learning_rate_main = 1e-3
#####################
# N = 400,  K = 300.0
# iterations_main = 50_000
# iterations_refinement = 4_000
# learning_rate_main = 1e-3
#####################
# N = 251, K = 250.5
# iterations_main = 30_000
# iterations_refinement = 2_000
# learning_rate_main = 1e-3
######################
# N = 152, K = 98.5
# iterations_main = 30_000
# iterations_refinement = 2_000
# learning_rate_main = 1e-3
######################
# N = 132, K = 90.5
# iterations_main = 25_000
# iterations_refinement = 2_000
# learning_rate_main = 5e-3
#######################
# N = 78, K = 73.0
# iterations_main = 10_000
# iterations_refinement = 1_000
# learning_rate_main = 1e-2

'''------------------------------- MAIN PART -------------------------------'''

if __name__ == "__main__":
    MIN_DIST = 0.5
    N = 78 # Number of sources
    K = 73.0 # normalized aperture length 
    MIN_JUMP = 1/K # Minimum gradient step.
    BEAM_WIDTH_FACTOR = 1.5# Beam width factor [1 .. 1.5] not more 1.5
    n_internal_elements = N - 2
    total_runs = 1 # Total run count
    iterations_main = 10_000 # number of iteration main stage gradient decent
    iterations_refinement = 1_000 # number of iteration stage refinement
    learning_rate_main = 1e-2 # step of 1 iteration GD

    all_slls = [] 
    all_configs = []
    global_best_sll = float('inf')

    print(f"\nWarming...")
    _, pts = optimize_single_run(4, 4, 10, 0.01)
    _, _ = cold_down_stage(pts, 4, 10, 0.01)

    time_start = time.time()
    for i in range(total_runs):
        print(f"[{i+1}]Stage 1: Global search: N={N}, Aperture={K} lambda")
        s_temp, p_temp = optimize_single_run(n_internal_elements, K, iterations_main, learning_rate_main)
        print(f'[{i+1}]Current PSLL:{10*np.log10(s_temp)}(dB).')

        print(f'[{i+1}]Stage 2: local refinement...')
        best_sll_val, best_pts = cold_down_stage(p_temp, K, iterations_refinement, learning_rate_main)

        all_slls.append(best_sll_val)
        all_configs.append(best_pts)
        
        if best_sll_val < global_best_sll:
            global_best_sll = best_sll_val

        print(f"[{i+1}]Best SLL: {10*np.log10(global_best_sll):.4f}(dB) |",
            f"Current SLL: {10*np.log10(best_sll_val):.4f}(dB)\n")

    time_one_run = (time.time() - time_start)/total_runs

    # Results aggregation
    aver_sll_db = 10*np.log10(np.mean(all_slls))
    best_idx = np.argmin(all_slls)
    min_sll_db = 10*np.log10(all_slls[best_idx])
    best_pts = all_configs[best_idx]
    print(f"\nMain search completed N = {N}, K = {K}.")
    print(f"For {total_runs} run(s):")
    print(f"Minimal SLL: {min_sll_db:.4f} dB")
    print(f"Average SLL: {aver_sll_db:.4f} dB")
    print(f"Average time: {time_one_run:.4f} sec")

    # if you need to print in terminal (this coordinates without 0 and 1)
    #print("Final coordinates:", best_pts.tolist())

    # Save the optimized configuration
    with open(f"coordinates_{N}_{K}.txt", "w") as f:
        f.write("0,\n")
        for p in best_pts:
            f.write(f"{p:.14f},\n")
        f.write("1\n")
    print(f'Coordinates saved to "coordinates_{N}_{K}.txt"')

    # Visualization
    plot_interference(best_pts, K, min_sll_db)
    input('Press "ENTER" to exit.')
