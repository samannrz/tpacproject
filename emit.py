import numpy as np
import matplotlib.pyplot as plt

# Parameters (from your TPAC setup)
f0 = 1e6                # frequency in Hz (1 MHz)
num_periods = 1
duty_cycle = 0.5
precision = 200         # samples per microsecond

# Convert precision to total number of samples
samples_per_period = int(precision)  # 200 samples per µs
T = 1 / f0              # period in seconds = 1 µs
total_time = num_periods * T

# Time vector in seconds
t = np.linspace(0, total_time, int(samples_per_period * total_time * 1e6), endpoint=False)

# Generate bipolar square wave
pulse = np.sign(np.sin(2 * np.pi * f0 * t))

# Plot
plt.plot(t * 1e6, pulse)
plt.title("Emitted Bipolar Pulse (1 MHz, 50% duty)")
plt.xlabel("Time (µs)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
