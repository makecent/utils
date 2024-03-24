from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

colors = plt.get_cmap("tab10").colors
markers = ['o', 'v', 's', 'd', 'X', 'p', '*', 'h', 'H']

x_values = [np.arange(1, 7), np.arange(1, 17)]
mean_aps = []

#%% figure 1
data = {
    # 'DITA': [29.19, 45.88, 57.47, 50.5, 60.69, 59.38, 63.22, 63.11, 62.47, 64.49, 64.24, 65.03, 65.51, 65.8, 65.63, 65.76],
    'DITA': [62.14, 64.45, 64.56, 65.84, 67.88, 68.24],
    # 'regression': [21.3, 19.8, 16.2, 15.2],
    # 'cost-sensitive': [21.1, 17.2, 15.5, 14.9],
    # 'binary decomposition': [20.3, 16.2, 15.2, 14.2],
    'TadTR': [33.31, 44.15, 42.69, 41.14, 49.09, 48.46, 52.69, 52.61, 52.93, 52.32, 52.97, 53.82, 52.86, 52.96, 54.92, 54.78],
}
fig, ax = plt.subplots(2, 1, height_ratios=[1, 1], figsize=plt.figaspect(1), layout='constrained')
for i, (name, value) in enumerate(data.items()):
    ln1_reg = ax[0].plot(x_values[i], value, label=name, color=colors[i], marker=markers[i])  # Plot some data on the axes.
# ax[0].set_xscale('log')
ax[0].legend()
ax[0].set_xticks(x_values[1], x_values[1])
# ax[0].xaxis.set_major_formatter(lambda x, pos: f'{x:.0f}' if x in [3, 6, 10, 20, 30] else None)
# ax[0].xaxis.set_minor_locator(ticker.MultipleLocator(1))
# ax[0].xaxis.set_minor_formatter(lambda x, pos: f'{x:.0f}' if x in [3, 6, 10, 20, 30] else None)
ax[0].set_xlabel('Epoch')
# ax[0].set_ylabel('Mean average precision (mAP)')
# fig.supxlabel(r'Number of ranks ($K$)')
fig.supylabel('mean average precision (mAP)')
ax[0].grid()
# plt.show()
#%% figure 2
x_values = (0.3, 0.4, 0.5, 0.6, 0.7, 'avg')
x = np.arange(len(x_values))
width = 0.45
data2 = {
    'DITA': [83.3, 78.7, 72.7, 60.3, 46.0, 68.2],
    'TadTR': [74.6, 67.4, 56.5, 44.9, 30.4, 54.8],
}
for i, (name, value) in enumerate(data2.items()):
    offset = width * i
    ln2_reg = ax[1].bar(x + offset, value, width=width, label=name, color=colors[i])
    ax[1].bar_label(ln2_reg, padding=3)
# ax[1].set_xscale('log')
# ax[1].set_xticks([3, 10, 30, 100], ['3', '10', '30', '100'])
ax[1].set_ylim(0, 120)
# ax[1].xaxis.set_major_formatter(lambda x, pos: f'{x:.0f}' if x in [10, 30, 100, 300] else None)
# ax[1].xaxis.set_minor_formatter(lambda x, pos: f'{x:.0f}' if x in [10, 30, 100, 300] else None)
ax[1].set_xlabel('IoU threshold')
ax[1].set_xticks(x + width/2, x_values)
# ax[1].set_ylabel('Mean average precision (mAP)')
ax[1].legend()
# ax[1].grid(zorder=3)
plt.show()
