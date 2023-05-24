from stress_corrected_asymptote import (
    ToughnessCorrectedDimless,
    ODEDimless,
    IntegralDimless,
    StressCorrectedAsymptote
)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time


plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = "\\usepackage{amsmath}"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = 18


# ========================================================================
# ========================================================================
#
# Dimensional form of the stress-corrected asymptote.
# ''Locating the fracture front'' example.
#
# ========================================================================
# ========================================================================

# Example of input parameters
e_prime = 1098901098.901099
k_prime = 3191538.2432114617
c_prime = 1e-14
mu_prime = 0.12
rho_layers = np.array([3, 5])
delta_sigma = np.array([-0.3e6, -0.2e6])

# Define the instances of solvers. options is optional parameter.
toughness_corrected_solver = StressCorrectedAsymptote(
    method='toughness-corrected', options={'n_stress_iter': 3}
)
ode_solver = StressCorrectedAsymptote(
    method='ode', options={'atol': 1e-8, 'rtol': 1e-5}
)
integral_solver = StressCorrectedAsymptote(
    method='integral', options={'n_stress_iter': 5, 'grid_n': 1000, 'is_adaptive_grid': True}
)

# ---------------------------------------------------------
# Compute fracture with
# ---------------------------------------------------------
# Define distance to the fracture tip and front velocity
distance = 6
velocity = 4.5

# Compute fracture width for the given distance and velocity using different methods
time_start = time.time()
width_toughness_corrected = toughness_corrected_solver.width(
    distance, velocity, rho_layers, delta_sigma, e_prime, k_prime, c_prime, mu_prime
)
time_toughness = time.time() - time_start
time_start = time.time()
width_ode = ode_solver.width(
    distance, velocity, rho_layers, delta_sigma, e_prime, k_prime, c_prime, mu_prime
)
time_ode = time.time() - time_start
time_start = time.time()
width_integral = integral_solver.width(
    distance, velocity, rho_layers, delta_sigma, e_prime, k_prime, c_prime, mu_prime
)
time_integral = time.time() - time_start
print("Dimensional fracture width at s = {s} meters:".format(s=distance))
print("    Toughness-corrected: w = {width} (computing time: {time})".format(
    width=width_toughness_corrected, time=time_toughness
))
print("    ODE approximation:   w = {width} (computing time: {time})".format(
    width=width_ode, time=time_ode
))
print("    Integral equation:   w = {width} (computing time: {time})".format(
    width=width_integral, time=time_integral
))

# ---------------------------------------------------------
# Compute distance to the fracture tip
# ---------------------------------------------------------
# Define fracture width, distance from the previous time step (see docs), and time step
width = 6e-3
s_old = 1.5
dt = 1

# Compute distance for the given width using different methods
time_start = time.time()
distance_toughness_corrected = toughness_corrected_solver.distance(
    width, s_old, dt, rho_layers, delta_sigma, e_prime, k_prime, c_prime, mu_prime
)
time_toughness = time.time() - time_start
time_start = time.time()
distance_ode = ode_solver.distance(
    width, s_old, dt, rho_layers, delta_sigma, e_prime, k_prime, c_prime, mu_prime
)
time_ode = time.time() - time_start
time_start = time.time()
distance_integral = integral_solver.distance(
    width, s_old, dt, rho_layers, delta_sigma, e_prime, k_prime, c_prime, mu_prime
)
time_integral = time.time() - time_start
print("Dimensional distance for w = {w} meters:".format(w=width))
print("    Toughness-corrected: s = {dist} (computing time: {time})".format(
    dist=distance_toughness_corrected, time=time_toughness
))
print("    ODE approximation:   s = {dist} (computing time: {time})".format(
    dist=distance_ode, time=time_ode
))
print("    Integral equation:   s = {dist} (computing time: {time})".format(
    dist=distance_integral, time=time_integral
))
print()

# ========================================================================
# ========================================================================
#
# Dimensionless stress-corrected solvers
#
# ========================================================================
# ========================================================================

fig_1 = plt.figure(figsize=(10, 8), tight_layout=True)
gs = gridspec.GridSpec(2, 2)

# Define the instances of asymptotes
toughness_corrected_asymptote = ToughnessCorrectedDimless(n_stress_iter=1)
ode_asymptote = ODEDimless()
integral_asymptote = IntegralDimless(is_adaptive_grid=True)

# Define the leak-off parameter, the stress layer location, and the amplitude of the stress barrier
chi = 1
layer_pos = 1e-3
delta_sigma = 161

# ---------------------------------------------------------
# Compute asymptote at the single point
# ---------------------------------------------------------
sp = 2e-3
w_toughness_sp = toughness_corrected_asymptote(sp, chi, layer_pos, delta_sigma)
w_ode_sp = ode_asymptote(sp, chi, layer_pos, delta_sigma)
w_integral_sp = integral_asymptote(sp, chi, layer_pos, delta_sigma)

print("Dimensionless width at s_tilde = {s}:".format(s=sp))
print("    Toughness-corrected:", w_toughness_sp)
print("    ODE approximation:  ", w_ode_sp)
print("    Integral equation:  ", w_integral_sp)


# ---------------------------------------------------------
# Compute asymptote at the prescribed array of points
# ---------------------------------------------------------
points = np.linspace(1e-8, 0.01, 100)

# Stress jump
# Note: To get the universal asymptotic solution,
# you should pass the empty array of the layer's positions and the amplitudes of the stress barriers.
w_universal_ap = toughness_corrected_asymptote(points, chi, [], [])
w_toughness_ap = toughness_corrected_asymptote(points, chi, layer_pos, delta_sigma)
w_ode_ap = ode_asymptote(points, chi, layer_pos, delta_sigma)
w_integral_ap = integral_asymptote(points, chi, layer_pos, delta_sigma)

ax_1 = fig_1.add_subplot(gs[0, 0])
ax_1.plot(points, w_universal_ap, color="#eda337", linewidth=2.5, linestyle='-', label="Universal Asymptote")
ax_1.plot(points, w_integral_ap, color="#191970", linewidth=2.5, linestyle='-', label="Integral Equation")
ax_1.plot(points, w_toughness_ap, color="#0b6623", linewidth=2.5, linestyle='--', label="Toughness Corrected")
ax_1.plot(points, w_ode_ap, color="#c81400", linewidth=2.5, linestyle='-.', label="ODE Approximation")

ax_1.set_title("Stress Jump")
ax_1.legend().set_draggable(True)
ax_1.grid(True)

# Stress drop
w_universal_ap = toughness_corrected_asymptote(points, chi, [], [])
w_toughness_ap = toughness_corrected_asymptote(points, chi, layer_pos, -delta_sigma)
w_ode_ap = ode_asymptote(points, chi, layer_pos, -delta_sigma)
w_integral_ap = integral_asymptote(points, chi, layer_pos, -delta_sigma)

ax_2 = fig_1.add_subplot(gs[0, 1])
ax_2.plot(points, w_universal_ap, color="#eda337", linewidth=2.5, linestyle='-', label="Universal Asymptote")
ax_2.plot(points, w_integral_ap, color="#191970", linewidth=2.5, linestyle='-', label="Integral Equation")
ax_2.plot(points, w_toughness_ap, color="#0b6623", linewidth=2.5, linestyle='--', label="Toughness Corrected")
ax_2.plot(points, w_ode_ap, color="#c81400", linewidth=2.5, linestyle='-.', label="ODE Approximation")

ax_2.set_title("Stress Drop")
ax_2.grid(True)

# ---------------------------------------------------------
# Two layers example
# ---------------------------------------------------------
layer_pos = [1e-3, 3e-3]
delta_sigma = [-80, 54]

w_universal_tl = toughness_corrected_asymptote(points, chi, [], [])
w_toughness_tl = toughness_corrected_asymptote(points, chi, layer_pos, delta_sigma)
w_ode_tl = ode_asymptote(points, chi, layer_pos, delta_sigma)
w_integral_tl = integral_asymptote(points, chi, layer_pos, delta_sigma)

ax_3 = fig_1.add_subplot(gs[1, :])
ax_3.plot(points, w_universal_tl, color="#eda337", linewidth=2.5, linestyle='-', label="Universal Asymptote")
ax_3.plot(points, w_integral_tl, color="#191970", linewidth=2.5, linestyle='-', label="Integral Equation")
ax_3.plot(points, w_toughness_tl, color="#0b6623", linewidth=2.5, linestyle='--', label="Toughness Corrected")
ax_3.plot(points, w_ode_tl, color="#c81400", linewidth=2.5, linestyle='-.', label="ODE Approximation")

ax_3.set_title("Drop/Jump")
ax_3.grid(True)

plt.show()
