from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
action_idx = {'BaseballPitch': 0, 'BasketballDunk': 1, 'Billiards': 2, 'CleanAndJerk': 3, 'CliffDiving': 4,
              'CricketBowling': 5, 'CricketShot': 6, 'Diving': 7, 'FrisbeeCatch': 8, 'GolfSwing': 9,
              'HammerThrow': 10, 'HighJump': 11, 'JavelinThrow': 12, 'LongJump': 13, 'PoleVault': 14, 'Shotput': 15,
              'SoccerPenalty': 16, 'TennisSwing': 17, 'ThrowDiscus': 18, 'VolleyballSpiking': 19}
action_idx_rev = {}
for key, val in action_idx.items():
    action_idx_rev[val] = key
indices = np.array(list(action_idx_rev.keys()))

tad = np.array([19.3, 38.5, 4.6, 54.1, 63.9, 15.1, 10.3, 26.9, 22.0, 20.5, 41.6, 22.0, 52.0, 71.7, 48.9, 16.0, 26.4, 12.3, 7.4, 10.8])
tad = np.append(tad, tad.mean())
scnn = np.array([16.0, 21.0, 7.5, 25.0, 27.5, 16.0, 14.0, 17.5, 16.0, 17.0, 19.0, 21.0, 16.5, 35.0, 31.5, 12.5, 19.0, 19.5, 25.0, 5.0])
scnn = np.append(scnn, scnn.mean())
apn = np.array([32.1, 49.3, 13.8, 66.7, 78.4, 34.2, 15.4, 77.2, 25.9, 77.7, 84.2, 86.4, 79.8, 92.4, 83.9, 60.9, 39.3, 19.5, 81.5, 46.5])
apn = np.append(apn, apn.mean())
x = np.arange(21*2, step=2)
plt.figure()
width = 0.5
plt.bar(x, tad, width=width, edgecolor='k', label='SS-TAD')
plt.bar(x+width, scnn, width=width, edgecolor='k', label='S-CNN')
plt.bar(x+width+width, apn, width=width, edgecolor='k', label='APN')

plt.xticks(ticks=x+width, labels=[action_idx_rev[ind] for ind in indices] + ['mAP'])
plt.gcf().set_size_inches(19.5/1.5, 4.5/1.5)
plt.gcf().set_dpi(1000)
plt.gcf().subplots_adjust(bottom=0.2)
for label in plt.gca().get_xticklabels()[:-1]:
    label.set_size(8)
    label.set_ha("right")
    label.set_rotation(30)
plt.gca().get_xticklabels()[-1].set_weight('bold')
# plt.gca().yaxis.set_minor_locator(MultipleLocator(5))
plt.gca().yaxis.set_major_locator(MultipleLocator(10))
plt.gca().yaxis.set_major_formatter('{x:.0f}')
plt.gca().tick_params(axis="y", direction='in', length=4, left=True, right=True)
plt.gca().tick_params(axis="x", direction='in', bottom=False)

plt.grid(axis='y', linestyle=':', which='both')
plt.gca().set_axisbelow(True)
plt.ylabel('Average Precision(%)')
plt.xlim([-1, 42])
plt.legend(loc='best')
plt.show()
