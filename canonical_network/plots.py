"""
=======================
Pie chart on polar axis
=======================

Demo of bar plot on a polar axis.
"""
import numpy as np
import matplotlib.pyplot as plt


# Compute pie slices
N = 128
bottom = 8
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
radii = -(4 - 3 * np.cos(theta / 2) ** 2)
# radii = -np.linspace(0.0, 4, N, endpoint=False)
radii[1] = -0.5
width = 2 * np.pi / N * np.ones(N)

# N = 80
# bottom = 8
# max_height = 4

# theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
# radii = max_height * np.random.rand(N)
# width = (2 * np.pi) / N

ax = plt.subplot(111, polar=True)
bars = ax.bar(theta, radii, width=width, bottom=bottom)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.grid(False)
# ax.set_theta_zero_location("N")

# Use custom colors and opacity
for r, bar in zip(radii, bars):
    bar.set_facecolor(plt.cm.viridis((r + 5.0) / 4.0))
    bar.set_alpha(0.8)

plt.show()
