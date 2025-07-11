import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ================================
# 1. Experimental Data (to be filled)
# ================================

# Pulse numbers (P): LTP and LTD separately
pulses_LTP = np.array([
    # Fill with pulse numbers for potentiation (e.g., [1, 2, 3, ..., N])
])
g_LTP = np.array([
    # Fill with conductance values corresponding to pulses_LTP
])
pulses_LTD = np.array([
    # Fill with pulse numbers for depression (e.g., [1, 2, 3, ..., N])
])
g_LTD = np.array([
    # Fill with conductance values corresponding to pulses_LTD
])

# Constants (known or from data)
G_min = None  # e.g., min conductance value
G_max = None  # e.g., max conductance value
P_max = None  # maximum number of pulses used in fitting

# ====================================
# 2. Define LTP and LTD Model Equations
# ====================================
# LTP Model: G(p) = B * (1 - exp(-P/A_LTP)) + G_min
def g_ltp_model(P, A_ltp):
    B = (G_max - G_min) / (1 - np.exp(-P_max / A_ltp))
    return B * (1 - np.exp(-P / A_ltp)) + G_min

# LTD Model: G(p) = -B * (1 - exp((P - Pmax)/A_LTD)) + G_max
def g_ltd_model(P, A_ltd):
    B = (G_max - G_min) / (1 - np.exp(-P_max / A_ltd))
    return -B * (1 - np.exp((P - P_max) / A_ltd)) + G_max

# ====================================
# 3. Fit the Curve to Extract A Values
# ====================================

# Initial guesses for A values
initial_guess_ltp = [None]  # Provide initial guess for A_ltp
initial_guess_ltd = [None]  # Provide initial guess for A_ltd

# Fit LTP
popt_ltp, _ = curve_fit(g_ltp_model, pulses_LTP, g_LTP, p0=initial_guess_ltp)
A_ltp_fit = popt_ltp[0]

# Fit LTD
popt_ltd, _ = curve_fit(g_ltd_model, pulses_LTD, g_LTD, p0=initial_guess_ltd)
A_ltd_fit = popt_ltd[0]

# ====================================
# 4. Plot the Results
# ====================================

# Plot LTP
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(pulses_LTP, g_LTP, label="LTP Data", color='blue')
plt.plot(pulses_LTP, g_ltp_model(pulses_LTP, A_ltp_fit), label=f"Fitted LTP\nA = {A_ltp_fit:.4f}", color='blue')
plt.xlabel("Pulse Number")
plt.ylabel("Conductance")
plt.title("LTP Fit")
plt.grid(True)
plt.legend()

# Plot LTD
plt.subplot(1, 2, 2)
plt.scatter(pulses_LTD, g_LTD, label="LTD Data", color='green')
plt.plot(pulses_LTD, g_ltd_model(pulses_LTD, A_ltd_fit), label=f"Fitted LTD\nA = {A_ltd_fit:.4f}", color='red')
plt.xlabel("Pulse Number")
plt.ylabel("Conductance")
plt.title("LTD Fit")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# ====================================
# 5. Display Results
# ====================================
print(f"Fitted A for LTP: {A_ltp_fit:.4f}")
print(f"Fitted A for LTD: {A_ltd_fit:.4f}")
# fin
#========================
# Find initial guess 
#========================
A_ltp_guess = (max(pulses_LTP) - min(pulses_LTP)) / 2
A_ltd_guess = (max(pulses_LTD) - min(pulses_LTD)) / 2
# fin
