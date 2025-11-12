# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 19:35:08 2025

@author: lexma
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# Define the Lorentzian function based on the provided equation
def lorentzian(omega, A, omega0, gamma):
    return A / np.sqrt(((omega0**2 - omega**2)**2 + (omega * gamma)**2))

# Function to read data from a file
def read_data(filename):
    data = np.loadtxt(filename, delimiter=',')  # Change delimiter if needed (e.g., '\t' for tab)
    omega_data = data[:, 0]  # First column (Frequency in rad/s)
    y_data = data[:, 1]  # Second column (Peak-to-Peak Voltage)
    y_uncertainties = data[:, 2]  # Third column (Uncertainties in Peak-to-Peak Voltage)
    return omega_data, y_data, y_uncertainties

# Input your data file name here
filename = 'n2data.txt'  # Change to your filename

# Read data
omega_data, y_data, y_uncertainties = read_data(filename)

# Square the y_data for fitting
y_data_squared = y_data**2
y_uncertainties_squared = 2 * y_data * y_uncertainties  # Propagate uncertainties

# Fit the data to the Lorentzian function
initial_guess = [np.max(y_data_squared), omega_data[np.argmax(y_data_squared)], 1]  # Initial guess for parameters
popt, pcov = curve_fit(lorentzian, omega_data, y_data_squared, p0=initial_guess, sigma=y_uncertainties_squared)

# Extract parameters
A, omega0, gamma = popt

# Calculate fitted values
y_fitted = lorentzian(omega_data, *popt)

# Calculate the half maximum
half_max = np.max(y_fitted) / 2

# Interpolate to find the FWHM
interp_func = interp1d(omega_data, y_fitted - half_max, bounds_error=False, fill_value="extrapolate")
omega_fwhm1 = interp_func(0)  # Find the first crossing
omega_fwhm2 = interp_func(0)  # Find the second crossing

# Find the frequencies corresponding to the half maximum
fwhm_freqs = omega_data[(y_fitted - half_max) * (np.roll(y_fitted, -1) - half_max) < 0]

# Calculate FWHM
if len(fwhm_freqs) >= 2:
    fwhm = fwhm_freqs[-1] - fwhm_freqs[0]
else:
    fwhm = np.nan  # If we can't find two crossing points

# Calculate chi-squared
chi_squared = np.sum(((y_data_squared - y_fitted) / y_uncertainties_squared) ** 2)
degrees_of_freedom = len(y_data_squared) - len(popt)
reduced_chi_squared = chi_squared / degrees_of_freedom if degrees_of_freedom > 0 else np.nan

# Print FWHM and chi-squared values
print("FWHM:", fwhm)
print("Chi-squared:", chi_squared)
print("Reduced Chi-squared:", reduced_chi_squared)

# Generate a smooth curve for the fitted Lorentzian
omega_fit = np.linspace(np.min(omega_data), np.max(omega_data), 1000)  # 1000 points for a smooth curve
y_fit = lorentzian(omega_fit, *popt)

# Plot the data and the fit
plt.figure(figsize=(10, 6))
plt.errorbar(omega_data, y_data_squared, yerr=y_uncertainties_squared, fmt='o', label='Data', markersize=5, color='blue', alpha=0.7)
plt.plot(omega_fit, y_fit, color='red', label='Fitted Lorentzian', linewidth=2)

# Correctly place the FWHM lines