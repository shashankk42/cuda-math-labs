import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV data
df = pd.read_csv('/path/to/roofline_demo.csv')

# Specify your hardware peaks (adjust to your GPU specs)
peak_bandwidth = 960.0  # GB/s (Taken from Nvidia Website, replace with your GPU's peak)
peak_flops = 91100.0    # GFLOP/s (Taken from Nvidia Website, replace with your GPU's peak)

# Prepare roofline curves
ai_min = df['AI'].min() / 2
ai_max = df['AI'].max() * 2
ai_curve = np.logspace(np.log10(ai_min), np.log10(ai_max), 200)
bw_roof = peak_bandwidth * ai_curve
compute_roof = np.full_like(ai_curve, peak_flops)

# Plot roofline
plt.figure()
plt.loglog(ai_curve, bw_roof, label=f'Bandwidth Ceiling ({peak_bandwidth} GB/s)')
plt.loglog(ai_curve, compute_roof, label=f'Compute Ceiling ({peak_flops} GFLOP/s)')
plt.scatter(df['AI'], df['GFLOP_s'], marker='o')

# Annotate kernels
for _, row in df.iterrows():
    plt.text(row['AI'], row['GFLOP_s'], row['Kernel'], fontsize=8, ha='right')

plt.xlabel('Arithmetic Intensity (FLOPs/Byte)')
plt.ylabel('Performance (GFLOP/s)')
plt.title('Roofline Analysis')
plt.legend()
plt.grid(True)
plt.savefig('roofline_plot.png')
# plt.show()
