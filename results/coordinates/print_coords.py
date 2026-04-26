import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def plot_interference(points, k_max, zoom_limit=0.15):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.bbox": "tight"
    })
    
    pts = np.sort(np.asarray(points))
    n_points = len(pts)
    m = np.linspace(0, k_max, 1_000_000) 
    u = m / k_max
    
    # Вместо m.reshape @ pts.reshape
    total_sum = np.zeros(len(m), dtype=np.complex128)
    for p in pts:
        total_sum += np.exp(2j * np.pi * m * p)
    intensity_db = 10 * np.log10(np.abs(total_sum)**2 / (n_points**2) + 1e-12)

    mask = u > (1.5 / k_max*MULT_MAIN_LOBE)
    sll_max_db = np.max(intensity_db[mask])

    fig, ax = plt.subplots(figsize=(7.1, 2.5))
    fig.set_dpi(300)

    ax.axhline(y=sll_max_db, color='#D62728', linestyle='--', lw=0.5, alpha=0.8,
                label=f'Peak PSL ({sll_max_db:.2f} dB)')
    ax.plot(u, intensity_db, color='#1F77B4', lw=0.5, label='Array Factor', rasterized=False)

    ax.set_xlabel('Normalized Direction $u = sin(\\theta)$')
    ax.set_ylabel('Intensity [dB]')
    ax.set_xlim(0, 1)
    ax.set_ylim(-50, 5)
    ax.grid(True, which='major', color='#EEEEEE', lw=0.5)

    ax.legend(loc='upper right', frameon=True, edgecolor='black', framealpha=1)
    plt.savefig('final_pattern_600.pdf', format='pdf', dpi=600)
    
    plt.show()

def select_configuration():
    print("="*60)
    print("   Sparse Array Optimization: High-Precision Framework")
    print("="*60)
    print("Available Configurations:")
    print(" [1] Case A:  152 elements on  98.5 lambda aperture")
    print(" [2] Case B:  132 elements on  90.5 lambda aperture")
    print(" [3] Case C:   78 elements on  73.0 lambda aperture")
    print(" [4] Case D:  251 elements on 250.5 lambda aperture")
    print(" [5] Large:   400 elements on 300.0 lambda aperture")
    print(" [6] Sat.:    600 elements on 450.0 lambda aperture")
    print(" [7] Over.:  1000 elements on 750.0 lambda aperture")
    print(" [8] Extreme:10000 elements on 7500.0 lambda aperture (Stress-test)")
    print("="*60)

    try:
        choice = int(input("Select configuration number (1-8): "))
        configs = {
            1: (152, 98.5),
            2: (132, 90.5),
            3: (78, 73.0),
            4: (251, 250.5),
            5: (400, 300.0),
            6: (600, 450.0),
            7: (1000, 750.0),
            8: (10000, 7500.0)
        }
        
        if choice in configs:
            N, nu = configs[choice]
            print(f"\n>>> Running Case {choice}: N={N}, Aperture={nu} lambda")
            print(f">>> This may take up to 30 seconds. Please wait...")
            return N, nu
        else:
            print("Invalid choice. Defaulting to Case C (78 elements).")
            return 78, 73.0
    except ValueError:
        print("Input error. Defaulting to Case C (78 elements).")
        return 78, 73.0
    
MULT_MAIN_LOBE = 1.5

N, k_val = select_configuration()

with open(f'coordinates_{N}_{k_val}.txt', 'r') as f:
    sources_list = [float(i) for i in f.read().split(',\n')]

plot_interference(sources_list, k_val)

