import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

action_idx = {'BaseballPitch': 0, 'BasketballDunk': 1, 'Billiards': 2, 'CleanAndJerk': 3, 'CliffDiving': 4,
              'CricketBowling': 5, 'CricketShot': 6, 'Diving': 7, 'FrisbeeCatch': 8, 'GolfSwing': 9,
              'HammerThrow': 10, 'HighJump': 11, 'JavelinThrow': 12, 'LongJump': 13, 'PoleVault': 14, 'Shotput': 15,
              'SoccerPenalty': 16, 'TennisSwing': 17, 'ThrowDiscus': 18, 'VolleyballSpiking': 19}
average_dur = {'Billiards': 3.462962962962963,
               'CleanAndJerk': 14.690476190476195,
               'CliffDiving': 2.941958041958043,
               'Diving': 2.9657314629258544,
               'CricketBowling': 1.5421348314606744,
               'CricketShot': 1.442541436464089,
               'FrisbeeCatch': 2.8155339805825226,
               'BaseballPitch': 3.67,
               'GolfSwing': 10.516129032258064,
               'HammerThrow': 7.112060301507539,
               'HighJump': 4.481549815498156,
               'JavelinThrow': 6.3598958333333355,
               'LongJump': 7.093865030674849,
               'PoleVault': 7.955833333333337,
               'Shotput': 4.971428571428572,
               'ThrowDiscus': 4.4724999999999975,
               'SoccerPenalty': 3.242424242424241,
               'BasketballDunk': 1.9726072607260738,
               'TennisSwing': 2.1826086956521746,
               'VolleyballSpiking': 1.9938356164383566}
tad = np.array(
    [19.3, 38.5, 4.6, 54.1, 63.9, 15.1, 10.3, 26.9, 22.0, 20.5, 41.6, 22.0, 52.0, 71.7, 48.9, 16.0, 26.4, 12.3, 7.4,
     10.8])
scnn = np.array(
    [16.0, 21.0, 7.5, 25.0, 27.5, 16.0, 14.0, 17.5, 16.0, 17.0, 19.0, 21.0, 16.5, 35.0, 31.5, 12.5, 19.0, 19.5, 25.0,
     5.0])
apn = np.array(
    [32.1, 49.3, 13.8, 66.7, 78.4, 34.2, 15.4, 77.2, 25.9, 77.7, 84.2, 86.4, 79.8, 92.4, 83.9, 60.9, 39.3, 19.5, 81.5,
     46.5])

# Sort actions by average duration
sorted_actions = sorted(average_dur, key=average_dur.get)
sorted_action_idx = {action: i for i, action in enumerate(sorted_actions)}

action_idx_rev = {}
for key, val in action_idx.items():
    action_idx_rev[val] = key
indices = np.array(list(action_idx_rev.keys()))

# Reorder tad, scnn, and apn arrays based on the sorted order
sorted_tad = np.array([tad[action_idx[action]] for action in sorted_actions])
sorted_scnn = np.array([scnn[action_idx[action]] for action in sorted_actions])
sorted_apn = np.array([apn[action_idx[action]] for action in sorted_actions])

# Append the mean for each method at the end
sorted_tad = np.append(sorted_tad, sorted_tad.mean())
sorted_scnn = np.append(sorted_scnn, sorted_scnn.mean())
sorted_apn = np.append(sorted_apn, sorted_apn.mean())

x = np.arange(21 * 2, step=2, dtype='float')
x[-1] += 0.5

# Adjust x for the sorted order
sorted_x = np.arange(len(sorted_actions) * 2, step=2, dtype='float')
sorted_x = np.append(sorted_x, sorted_x[-1] + 2.5)  # Adjust for "All actions"

# Plot
figsize = plt.figaspect(4.5 / 19.5)
fig, ax = plt.subplots(layout='constrained', figsize=figsize, dpi=800)
# fig.set_dpi(1000)

width = 0.5
ax.bar(x, sorted_tad, width=width, edgecolor='k', label='SS-TAD')
ax.bar(x + width, sorted_scnn, width=width, edgecolor='k', label='S-CNN')
ax.bar(x + width + width, sorted_apn, width=width, edgecolor='k', label='APN')

#%% Add average duration to the xtick labels, formatting the duration to keep two numbers after the decimal point
xtick_labels_with_duration = [f"{action}\n({average_dur[action]:.2f}s)" for action in sorted_actions] + ['All actions']
ax.set_xticks(ticks=sorted_x + width, labels=xtick_labels_with_duration)

#%% Add text boxes on top of specific bars
texts_to_add = {
    "HighJump": "Progressive",
    "CricketShot": "Ephemeral",
    "Billiards": "Indeterminate",
    # Add more actions and texts as needed
    # "AnotherAction": "Text"
}

# Loop through the dictionary to add texts to the specified bars
for action, text in texts_to_add.items():
    if action in sorted_actions:
        # Find the index of the action in sorted_actions to determine its position on the x-axis
        action_index = sorted_actions.index(action)
        action_x_pos = sorted_x[action_index] + width  # Adjust based on bar width and position

        # Find the maximum value among the methods for the action to determine the y position for the text
        action_y_pos = max(sorted_tad[action_index], sorted_scnn[action_index], sorted_apn[action_index])

        # Add the specified text above the bar for the action
        ax.text(action_x_pos, action_y_pos + 2, text, ha='center', va='bottom', fontsize=8, fontweight='bold')
        # The '+ 2' in the y position is to offset the text above the bar, adjust as needed


#%% Rotate the x ticks
for label in plt.gca().get_xticklabels()[:-1]:
    label.set_size(8)
    label.set_ha("right")
    label.set_rotation(30)

#%% Miscellaneous settings
ax.get_xticklabels()[-1].set_weight('bold')
# plt.gca().yaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_major_formatter('{x:.0f}')
ax.tick_params(axis="y", direction='in', length=4, left=True, right=True)
ax.tick_params(axis="x", direction='in', bottom=False)

plt.grid(axis='y', linestyle=':', which='both')
ax.set_axisbelow(True)
ax.set_ylabel('Average Precision (%, IoU=0.5)')
ax.set_xlim([-1, len(sorted_actions) * 2 + 2.5])
ax.annotate('mAP', (x[-1], 60), fontweight='bold')
ax.legend(loc='upper left')
fig.show()
