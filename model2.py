import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ===========================
# 1. Insert Experimental Data
# ===========================

pulses_LTP = np.array([
    # Fill pulse numbers for potentiation
])

g_LTP = np.array([
    # Corresponding conductance values for LTP
])

pulses_LTD = np.array([
    # Fill pulse numbers for depression
])

g_LTD = np.array([
    # Corresponding conductance values for LTD
])

# ========================================
# 2. Define Constants Based on Your Data
# ========================================

G_min = None  # e.g., np.min(g_LTP)
G_max = None  # e.g., np.max(g_LTP)
N = None      # Total number of pulses used (e.g., np.max(pulses_LTP))

# ========================================
# 3. Define Model Equations for Fitting
# ========================================

def ltp_model(n_p, k):
    A = (G_max - G_min) * (1 + np.exp(k) / N)
    return G_min + A * (n_p / (n_p + np.exp(k)))

def ltd_model(n_d, k):
    A = (G_max - G_min) * (1 + np.exp(k) / N)
    return G_max - A * ((N - n_d) / ((N - n_d) + np.exp(k)))

# ========================================
# 4. Provide Initial Guess and Fit
# ========================================

initial_guess_k = [None]  # Provide a reasonable estimate (e.g., log(1) or log(N/2))

# Fit LTP
popt_ltp, _ = curve_fit(ltp_model, pulses_LTP, g_LTP, p0=initial_guess_k)
k_ltp_fit = popt_ltp[0]

# Fit LTD
popt_ltd, _ = curve_fit(ltd_model, pulses_LTD, g_LTD, p0=initial_guess_k)
k_ltd_fit = popt_ltd[0]

# ========================================
# 5. Calculate ANL from Fitted Curves
# ========================================

n_half = N / 2
G_p_half = ltp_model(n_half, k_ltp_fit)
G_d_half = ltd_model(n_half, k_ltd_fit)

ANL = (G_p_half - G_d_half) / (G_max - G_min)

# ========================================
# 6. Plotting (Optional)
# ========================================

plt.figure(figsize=(10, 5))

# LTP
plt.subplot(1, 2, 1)
plt.scatter(pulses_LTP, g_LTP, label="LTP Data", color='blue')
plt.plot(pulses_LTP, ltp_model(pulses_LTP, k_ltp_fit), label=f"LTP Fit\nk = {k_ltp_fit:.4f}", color='red')
plt.title("LTP Fit")
plt.xlabel("Pulse")
plt.ylabel("Conductance")
plt.grid(True)
plt.legend()

# LTD
plt.subplot(1, 2, 2)
plt.scatter(pulses_LTD, g_LTD, label="LTD Data", color='green')
plt.plot(pulses_LTD, ltd_model(pulses_LTD, k_ltd_fit), label=f"LTD Fit\nk = {k_ltd_fit:.4f}", color='orange')
plt.title("LTD Fit")
plt.xlabel("Pulse")
plt.ylabel("Conductance")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# ========================================
# 7. Output Results
# ========================================

print(f"Fitted k for LTP: {k_ltp_fit:.4f}")
print(f"Fitted k for LTD: {k_ltd_fit:.4f}")
print(f"ANL: {ANL:.4f}")
