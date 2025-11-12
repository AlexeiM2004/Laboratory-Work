# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 23:56:06 2024

@author: lexma
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import G

# Constants
SPEED_OF_LIGHT = 3.0e8  # Speed of light in m/s
SOLAR_MASS = 1.989e+30  # Solar mass in kg
MASS_SOLAR = 2.78 * SOLAR_MASS
EMITTED_WAVELENGTH = 656.281e-9  # Emitted wavelength in m
AU_IN_METRES = 1.496e+11  # AU in metres
JOVIAN_MASS = 1.898e+27  # Jovian Mass in kg

def get_inclination_angle():
    """Function to prompt the user for the angle of inclination."""
    while True:
        try:
            angle = float(input("Please enter the angle of inclination (in degrees): "))
            if angle == 0 or math.isclose(angle % 180, 0):
                print("Invalid input. Angle of inclination cannot be zero or a multiple of pi.")
            else:
                return angle
        except ValueError:
            print("Invalid input. Please enter a valid angle of inclination.")

# Get the angle of inclination from the user
angle_of_inclination = get_inclination_angle()

# Convert the angle from degrees to radians
angle_of_inclination_rad = np.radians(angle_of_inclination)

def wavelength_to_velocity(wavelength, wavelength_uncertainty):
    """Function to convert wavelength to velocity."""
    wavelength_m = wavelength * 1e-9  # Convert wavelength from nm to m
    # Convert wavelength uncertainty from nm to m
    wavelength_uncertainty_m = wavelength_uncertainty * 1e-9
    velocity = ((wavelength_m * SPEED_OF_LIGHT / EMITTED_WAVELENGTH) -
                SPEED_OF_LIGHT) / abs(np.sin(angle_of_inclination_rad))
    velocity_uncertainty = (wavelength_uncertainty_m * SPEED_OF_LIGHT) / (EMITTED_WAVELENGTH)
    return velocity, velocity_uncertainty

def model(params, t):
    """Model function."""
    v0, omega, phi = params
    return v0 * np.sin(omega * t + phi)

def read_data(file_name):
    """Function to read data from a file."""
    return np.genfromtxt(file_name, delimiter=',')

def concatenate_data(data1, data2):
    """Function to concatenate data."""
    return np.concatenate((data1, data2))

def remove_nan_values(data):
    """Function to remove rows with non-numerical or NaN values."""
    clean_data = []
    for _, row in enumerate(data):
        if row[2] != 0:
            clean_data.append(row)  # to remove 0 values
    data = np.array(clean_data)
    return data[~np.isnan(data).any(axis=1)]


def anomaly_detection(tolerance, data):
    """Function to detect anomalies."""
    removed_anomalies = [data[0], data[len(data)-1]]
    for counter in range(1, len(data)-1):
        prev_diff = abs(data[counter][1] - data[counter-1][1])
        next_diff = abs(data[counter][1] - data[counter+1][1])
        if next_diff < tolerance or prev_diff < tolerance:
            removed_anomalies.append(data[counter])
    return np.array(removed_anomalies)

def v_s_model(t, v_0, omega):
    """Function to define v_s_model."""
    return v_0 * np.sin(omega*t + np.pi)

# Read data
DATA1 = read_data('doppler_data_1.csv')
DATA2 = read_data('doppler_data_2.csv')

# Concatenate data
DATA = concatenate_data(DATA1, DATA2)

# Remove rows with non-numerical or NaN values
DATA = remove_nan_values(DATA)

DATA = anomaly_detection(0.00005, DATA)

# Separate data into different arrays
TIME = DATA[:, 0] * 365.25 * 24 * 60 * 60  # Convert time from years to seconds
WAVELENGTH = DATA[:, 1]
WAVELENGTH_UNCERTAINTY = DATA[:, 2]

# Convert wavelength to velocity
VELOCITY, VELOCITY_UNCERTAINTY = wavelength_to_velocity(WAVELENGTH, WAVELENGTH_UNCERTAINTY)

# Initial guesses for parameters
INITIAL_GUESSES = [50, 3*10**-8]

# Sort data for plotting
SORTED_INDICES = np.argsort(TIME)
SORTED_T = TIME[SORTED_INDICES]
SORTED_V_S_VALUES = VELOCITY[SORTED_INDICES]

# Fit model to data
POPT, PCOV, *_ = curve_fit(v_s_model, TIME, VELOCITY, p0=INITIAL_GUESSES,
                           sigma = VELOCITY_UNCERTAINTY, full_output = True)
PERR = np.sqrt(np.diag(PCOV))
ABSANGVEL = np.abs(PCOV[0][1])

# Calculate minimized chi-squared
RESIDUALS =SORTED_V_S_VALUES - v_s_model(SORTED_T, POPT[0], POPT[1])
CHI_SQUARED = np.sum((RESIDUALS / VELOCITY_UNCERTAINTY)**2)
# Calculate reduced chi-squared
REDUCED_CHI_SQUARED = CHI_SQUARED / (len(DATA) - 2)

# Print the fitted parameters
print(f"v0 = {POPT[0]:.4g} ± {PCOV[0][0]:.2g} m/s")
print(f"Omega = {POPT[1]:.4g} ± {ABSANGVEL:.2g} rad/s")

# Print the minimized chi-squared and reduced chi-squared
print("Chi-squared:", CHI_SQUARED)
print("Reduced Chi-squared:", REDUCED_CHI_SQUARED)

STAR_DISTANCE = (((G*MASS_SOLAR)/(POPT[1]**2))**(1/3))/AU_IN_METRES #  in AU
PLANET_VELOCITY = ((G*MASS_SOLAR)/(STAR_DISTANCE*AU_IN_METRES))**(1/2) # in m/s
PLANET_MASS = ((MASS_SOLAR*POPT[0])/(PLANET_VELOCITY))/(JOVIAN_MASS) # in kg
CHANGE_PERIOD = (-2*np.pi*abs(PCOV[0][1]))/(POPT[1])**2
                #Calculates the uncertainty in period
CHANGE_DISTANCE = abs(((G*MASS_SOLAR)/((3*POPT[1])*((G*MASS_SOLAR)/(POPT[1])**2)
                                       **(2/3))*(CHANGE_PERIOD))/AU_IN_METRES)
                                        #Calculates uncertainty in distance
CHANGE_VELOCITY_PLANET = (G*MASS_SOLAR)/(2*(STAR_DISTANCE*AU_IN_METRES)**2
                                         *((G*MASS_SOLAR)/(STAR_DISTANCE*AU_IN_METRES))**2)
                                        #Calculates uncertainty in planet velocity
CHANGE_MASS_PLANET = PLANET_MASS*((CHANGE_VELOCITY_PLANET/PLANET_VELOCITY)
                                  **2+(PCOV[0][0]/POPT[0])**2)**(1/2)
                                        #Calculates uncertainty in planet mass
print(f"Star Distance = {STAR_DISTANCE:.4g} ± {CHANGE_DISTANCE:.2g} AU")
print(f"Planet Mass = {PLANET_MASS:.4g} ± {CHANGE_MASS_PLANET:.2g} Jovian Masses")

# Plot results
plt.figure(figsize=(8,6))
plt.errorbar(SORTED_T, SORTED_V_S_VALUES, yerr=VELOCITY_UNCERTAINTY, fmt='o',
             capsize=5, label='Observed Velocities')  # Plot error bars
plt.plot(SORTED_T, v_s_model(SORTED_T, *POPT), color ='red', label='Expected Velocities')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (ms/1)')
plt.title('Velocity (m/s) vs Time (s)')
plt.legend()
plt.grid(True)

# Add parameter values below the plot
plt.figtext(0.1, -0.15, f"v0 = {POPT[0]:.4g} ± {PCOV[0][0]:.2g} m/s",
            transform=plt.gca().transAxes)
plt.figtext(0.5, -0.15, f"Omega = {POPT[1]:.4g} ± {ABSANGVEL:.2g} rad/s",
            transform=plt.gca().transAxes)
plt.figtext(0.1, -0.25, f"Chi-squared: {CHI_SQUARED:.5g}",
            transform=plt.gca().transAxes)
plt.figtext(0.5, -0.25, f"Reduced Chi-squared: {REDUCED_CHI_SQUARED:.5g}",
            transform=plt.gca().transAxes)
plt.figtext(0.1, -0.35, f"Star Distance = {STAR_DISTANCE:.4g} ± {CHANGE_DISTANCE:.2g} AU",
            transform=plt.gca().transAxes)
plt.figtext(0.5, -0.35, f"Planet Mass = {PLANET_MASS:.4g} ± {CHANGE_MASS_PLANET:.2g} Jovian Masses",
            transform=plt.gca().transAxes)

plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin to make room for the parameter values

plt.show()
