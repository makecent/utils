from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

colors = plt.get_cmap("tab10").colors
markers = ['o', 'v', 's', 'd', 'X', 'p', '*', 'h', 'H']

k_values = [3, 6, 10, 20, 30]
mean_aps = []

#%% mAP figure
mAP = {
    'classification': [32.7, 34.9, 35.1, 36.1, 36.8],
    # 'regression': [21.3, 19.8, 16.2, 15.2],
    # 'cost-sensitive': [21.1, 17.2, 15.5, 14.9],
    # 'binary decomposition': [20.3, 16.2, 15.2, 14.2],
    'ordinal regression': [35.7, 37.0, 37.6, 38.4, 38.8],
}
fig, ax = plt.subplots(1, 2, figsize=plt.figaspect(0.7), layout='constrained')
for i, (name, value) in enumerate(mAP.items()):
    ln1_reg = ax[0].plot(k_values, value, label=name, color=colors[i], marker=markers[i])  # Plot some data on the axes.
# ax[0].set_xscale('log')
ax[0].legend()
# ax[0].set_xticks([3, 10, 30, 100], ['3', '10', '30', '100'])
ax[0].xaxis.set_major_formatter(lambda x, pos: f'{x:.0f}' if x in [3, 6, 10, 20, 30] else None)
ax[0].xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax[0].xaxis.set_minor_formatter(lambda x, pos: f'{x:.0f}' if x in [3, 6, 10, 20, 30] else None)
# ax[0].set_xlabel('Number of ranks')
# ax[0].set_ylabel('Mean average precision (mAP)')
fig.supxlabel(r'Number of ranks ($K$)')
fig.supylabel('Mean average precision (mAP)')
ax[0].grid()

#%% input resolution
k_values = [10, 30, 100, 300]
mAP = {
    '8   x 4 ': [27.6, 28.8, 28.7, 28.2],
    '16 x 4': [34.2, 34.5, 34.5, 34.2],
    '32 x 4': [37.3, 37.8, 38.4, 38.8],
}
for i, (name, value) in enumerate(mAP.items()):
    ln1_reg = ax[1].plot(k_values, value, label=name, color=colors[i+1], marker=markers[i+1])  # Plot some data on the axes.
ax[1].set_xscale('log')
# ax[1].set_xticks([3, 10, 30, 100], ['3', '10', '30', '100'])
ax[1].xaxis.set_major_formatter(lambda x, pos: f'{x:.0f}' if x in [10, 30, 100, 300] else None)
ax[1].xaxis.set_minor_formatter(lambda x, pos: f'{x:.0f}' if x in [10, 30, 100, 300] else None)
# ax[1].set_xlabel('Number of ranks')
# ax[1].set_ylabel('Mean average precision (mAP)')
ax[1].legend(loc=(0.65, .3))
ax[1].grid()
plt.show()