import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats


# ----------------------------- #
#    Spring: Hooke’s Law Fit    #
# ----------------------------- #


# Constants
g = 9.81


# Load data
spring_df = pd.read_csv('spring-project/data/spring_defl_vs_added_mass.csv')


# Convert to SI units
spring_df["spring_deflection_m"] = spring_df["spring_deflection_cm"] / 100
spring_df["added_mass_kg"] =  spring_df["added_mass_g"] / 1000

# Accuracy of the measuring instrument in (meters)
accuracy = 0.001

# Measurement uncertanity of the measuring instrument (in meters)
spring_df["u_x"] = accuracy / np.sqrt(3)


# Definition of model function: x = A0 * m + A1
def linear_model(m, A0, A1):
    return A0 *m + A1


# First fit
initial_guesses = (0.5, 0.5)
popt, pcov = curve_fit(
    linear_model, spring_df["added_mass_kg"], spring_df["spring_deflection_m"], 
    sigma = spring_df["u_x"], absolute_sigma=True, p0=initial_guesses
)
A0, A1 = popt
std_errors = np.sqrt(np.diag(pcov))

print("Initial fit:")
print(f"Fitted A0 = {A0:.5f} +/- {std_errors[0]:.5f}")
print(f"Fitted A1 = {A1:.5f} +/- {std_errors[1]:.5f}")

# Plot first fit
m_vals = np.linspace(np.min(spring_df["added_mass_kg"]), np.max(spring_df["added_mass_kg"]),100)
x_vals = linear_model(m_vals, A0, A1)

plt.errorbar(spring_df["added_mass_kg"], spring_df["spring_deflection_m"], spring_df["u_x"], 
             markersize=2, fmt='o', color="black", label = "Measurements")
plt.plot(m_vals, x_vals, 'r-', label = "First fit")
plt.xlabel('Added mass [kg]')
plt.ylabel('Spring deflection $\Delta$x [m]')
plt.title('Spring deflection vs added mass')
plt.legend()
plt.savefig("spring-project/figures/fit_before_scaling.png", dpi=300)
plt.show()


# chi-squared test
residuals = spring_df['spring_deflection_m'] - linear_model(spring_df['added_mass_kg'], A0, A1)
chi_squared = np.sum((residuals/ spring_df['u_x'])**2)
ndof = len(spring_df) - len(popt) # number of degrees of freedom
reduced_chi_squared = chi_squared / ndof
statistical_significance = 0.05
critical_chi_squared = stats.distributions.chi2.isf(statistical_significance, ndof)

print(f"Chi^2 = {chi_squared:.2f}")
print(f"Chi^2 (reduced) = {reduced_chi_squared:.2f}")
print(f"Chi^2 (critical) = {critical_chi_squared:.2f}")


# If reduced chi-squared > 1.5, rescale uncertanities
chi2_threshold = 1.5
if reduced_chi_squared > chi2_threshold:
    
    scale_factor = np.sqrt(reduced_chi_squared)
    spring_df["u_x"] = spring_df["u_x"] * scale_factor

    # Fit again
    popt, pcov = curve_fit(
        linear_model, spring_df["added_mass_kg"], spring_df["spring_deflection_m"],
        sigma = spring_df["u_x"], absolute_sigma=True, p0=initial_guesses
    )
    A0, A1 = popt
    std_errors = np.sqrt(np.diag(pcov))
    print("\nSecond fit:")
    print(f"Fitted A0 = {A0:.5f} +/- {std_errors[0]:.5f}")
    print(f"Fitted A1 = {A1:.5f} +/- {std_errors[1]:.5f}")

    # Plot after scaling
    x_vals = linear_model(m_vals, A0,A1)

    plt.errorbar(spring_df["added_mass_kg"], spring_df["spring_deflection_m"], spring_df["u_x"], 
                 markersize=2, fmt='o', color="black", label = "Measurements")
    plt.plot(m_vals, x_vals, 'b-', label = "Fit afer scaling")
    plt.xlabel('Added mass [kg]')
    plt.ylabel('Spring deflection $\Delta$x [m]')
    plt.title('Spring deflection vs added mass')
    plt.legend()
    plt.savefig("spring-project/figures/fit_after_scaling.png", dpi=300)
    plt.show()


# Compute spring constant k and its uncertanity u_k
k = g / A0
u_k = g * std_errors[0] / A0**2

print("\nSpring constant:")
print(f"k = {k:.2f} N/m +/- {u_k:.2f} N/m")