import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =======================================
# 1. Provide Experimental Data
# =======================================

pulses_LTP = np.array([
    # Pulse numbers for LTP
])

g_LTP = np.array([
    # Corresponding conductance values for LTP
])

pulses_LTD = np.array([
    # Pulse numbers for LTD
])

g_LTD = np.array([
    # Corresponding conductance values for LTD
])

# =======================================
# 2. Define Constants from Data
# =======================================

G_min = None  # e.g., np.min(g_LTD)
G_max = None  # e.g., np.max(g_LTP)
N = None      # Total number of pulses

ΔG = G_max - G_min

# =======================================
# 3. Define Fitting Models
# =======================================

# LTP: G = Gmax - beta_p * exp(-alpha_p * n)
def ltp_model(n, beta_p, alpha_p):
    return G_max - beta_p * np.exp(-alpha_p * n)

# LTD: G = Gmin + beta_d * exp(-alpha_d * n)
def ltd_model(n, beta_d, alpha_d):
    return G_min + beta_d * np.exp(-alpha_d * n)

# =======================================
# 4. Fit Models to Extract Parameters
# =======================================

initial_guess_ltp = [None, None]  # [beta_p, alpha_p]
initial_guess_ltd = [None, None]  # [beta_d, alpha_d]

popt_ltp, _ = curve_fit(ltp_model, pulses_LTP, g_LTP, p0=initial_guess_ltp)
beta_p, alpha_p = popt_ltp

popt_ltd, _ = curve_fit(ltd_model, pulses_LTD, g_LTD, p0=initial_guess_ltd)
beta_d, alpha_d = popt_ltd

# =======================================
# 5. Calculate ANL, DANL, PANL
# =======================================

n_half = N / 2
G_p_half = ltp_model(n_half, beta_p, alpha_p)
G_d_half = ltd_model(n_half, beta_d, alpha_d)

DANL = 0.5 - (G_d_half / ΔG)
PANL = (G_p_half / ΔG) - 0.5
ANL = 1 - (beta_p * np.exp(-0.5 * alpha_p * N) + beta_d * np.exp(-0.5 * alpha_d * N)) / ΔG

# =======================================
# 6. Plot Fitted Results (Optional)
# =======================================

plt.figure(figsize=(10, 5))

# LTP Plot
plt.subplot(1, 2, 1)
plt.scatter(pulses_LTP, g_LTP, label='LTP Data', color='blue')
plt.plot(pulses_LTP, ltp_model(pulses_LTP, beta_p, alpha_p), label=f'Fit (α={alpha_p:.4f})', color='red')
plt.title('LTP Fit')
plt.xlabel('Pulse Number')
plt.ylabel('Conductance')
plt.grid(True)
plt.legend()

# LTD Plot
plt.subplot(1, 2, 2)
plt.scatter(pulses_LTD, g_LTD, label='LTD Data', color='green')
plt.plot(pulses_LTD, ltd_model(pulses_LTD, beta_d, alpha_d), label=f'Fit (α={alpha_d:.4f})', color='orange')
plt.title('LTD Fit')
plt.xlabel('Pulse Number')
plt.ylabel('Conductance')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# =======================================
# 7. Output Final Values
# =======================================

print(f"Fitted Parameters:")
print(f"  LTP: beta_p = {beta_p:.4f}, alpha_p = {alpha_p:.4f}")
print(f"  LTD: beta_d = {beta_d:.4f}, alpha_d = {alpha_d:.4f}")
print()
print(f"DANL = {DANL:.4f}")
print(f"PANL = {PANL:.4f}")
print(f"Total ANL = {ANL:.4f}")
