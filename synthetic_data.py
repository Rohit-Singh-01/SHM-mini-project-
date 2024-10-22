import numpy as np
import pandas as pd
import datetime

# Parameters
n_samples = 1000  # Number of data points
start_time = datetime.datetime.now()
time_interval = datetime.timedelta(seconds=1)  # 1-second intervals

# Generate timestamps
timestamps = [start_time + i * time_interval for i in range(n_samples)]

# Generate synthetic sensor data
np.random.seed(42)

# Simulate healthy structure data
vibration_healthy = np.random.normal(loc=0, scale=1, size=n_samples)
strain_healthy = np.random.normal(loc=0.1, scale=0.02, size=n_samples)
displacement_healthy = np.random.normal(loc=0.5, scale=0.05, size=n_samples)
temperature = np.random.normal(loc=25, scale=1, size=n_samples)  # Constant temp

# Simulate damaged structure data
vibration_damaged = np.random.normal(loc=5, scale=2, size=n_samples)
strain_damaged = np.random.normal(loc=0.3, scale=0.05, size=n_samples)
displacement_damaged = np.random.normal(loc=2, scale=0.2, size=n_samples)

# Combine healthy and damaged data
condition = ['healthy'] * (n_samples // 2) + ['damaged'] * (n_samples // 2)
vibration = np.concatenate([vibration_healthy[:n_samples // 2], vibration_damaged[n_samples // 2:]])
strain = np.concatenate([strain_healthy[:n_samples // 2], strain_damaged[n_samples // 2:]])
displacement = np.concatenate([displacement_healthy[:n_samples // 2], displacement_damaged[n_samples // 2:]])

# Create DataFrame
data = pd.DataFrame({
    'timestamp': timestamps,
    'vibration': vibration,
    'strain': strain,
    'displacement': displacement,
    'temperature': temperature,
    'condition': condition
})

# Save the synthetic data to a CSV file
data.to_csv('synthetic_shm_data.csv', index=False)
print("Synthetic data generated and saved to 'synthetic_shm_data.csv'")
