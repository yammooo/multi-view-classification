import pandas as pd
import numpy as np

# Hardcoded CSV file path (adjust as needed)
csv_file = "/home/yammo/Development/multi-view-classification/gpu_power_fusions.csv"

# Read the CSV. This example assumes your CSV has one time column and all other columns are power (in Watts).
df = pd.read_csv(csv_file)

# Define the time column name (adjust if necessary)
time_col = "Relative Time (Process)"
if time_col not in df.columns:
    raise ValueError(f"Expected time column '{time_col}' not found. Available columns: {df.columns.to_list()}")

# Convert time column to numeric and sort by time
df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
df = df.sort_values(time_col).dropna(subset=[time_col])

# Identify power columns (all columns except the time column)
power_cols = [col for col in df.columns if col != time_col]

# Prepare a dictionary to hold results
energy_results = {}

# Loop through each power column
for col in power_cols:
    # Convert power data to numeric
    df[col] = pd.to_numeric(df[col], errors='coerce')
    # Keep only rows where both time and power are valid
    valid = df[[time_col, col]].dropna()
    # Compute time differences (s). Assume time is in seconds.
    dt = valid[time_col].diff().fillna(0)
    # Total energy (Joules): sum(Watts * seconds)
    energy_joules = np.sum(valid[col] * dt)
    # Convert Joules to kilowatt-hours: 1 kWh = 3.6e6 Joules
    energy_kwh = energy_joules / 3.6e6
    energy_results[col] = (energy_joules, energy_kwh)

# Print the results
print("Total energy consumption per GPU channel:")
for col, (joules, kwh) in energy_results.items():
    print(f" {col}: {joules:.1f} Joules, or {kwh:.4f} kWh")