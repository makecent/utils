import matplotlib.pyplot as plt
import numpy as np

# Your data
data = {
    'Baseline': 62.4,
    'tFPN': 64.5,
    'Unsup.': 67.0,
    'MF': 67.6,
    'DN': 69.2,
    'SQR': 69.4,
    'GQG': 70.4,
}

# Preparing data for plotting
techniques = list(data.keys())
values = list(data.values())

# Creating the plot
fig, ax1 = plt.subplots(figsize=(8, 6), layout='constrained')  # Reduced figure width for a more compact x-axis

# Adjusting the width of bars to make them narrower, effectively reducing the space between points
bar_width = 0.7
bars = ax1.bar(np.arange(len(techniques)), values, color='lightblue', width=bar_width)

# Adjusting the line plot to align with the updated bar positions
ax1.plot(np.arange(len(techniques)), values, color='darkblue', marker='o', linestyle='-', linewidth=2, markersize=8)

# Adjusting y-axis and setting y-ticks with an appropriate range and interval
ax1.set_ylim(bottom=60, top=71)
ax1.set_yticks(np.arange(60, max(values) + 1, 2))

# Updating y-axis title
ax1.set_ylabel('Average Mean Average Precision (avg. mAP)', fontsize=14)

# Updating x-axis ticks to include "+" prefix for techniques other than "Baseline" and aligning them with bars
ax1.set_xticks(np.arange(len(techniques)))
ax1.set_xticklabels([technique if technique == 'Baseline' else f'+{technique}' for technique in techniques], fontsize=12)

# Enabling grid lines for better readability
ax1.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)

# Updating the overall title to "Ablation Study"
plt.title('Ablation Study', fontsize=16)

plt.show()

