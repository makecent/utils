import matplotlib.pyplot as plt
from intervaltree import Interval, IntervalTree
import mmengine
from mmengine.utils import track_iter_progress

data = mmengine.load(r"../assets/tad_annotations/ANet/anet_val.json")

# Analyze the data
# 1. Analyze the video duration, the min/max/mean/meadian of the video duration
# 2. Analyze the number of segments, the min/max/mean/median of the number of segments
# 3. Analyze the duration of segments, the min/max/mean/median of the duration of segments
# 4. Analyze the percentage of the duration of action in the video duration, the min/max/mean/median.
# 5. Plot the boxplot of all results

video_duration = []
num_segments = []
duration_segments = []
percentage_actions = []
for video_name, video_info in track_iter_progress(data.items()):
    video_duration.append(video_info['duration'])
    num_segments.append(len(video_info['segments']))
    segments = list(filter(lambda x: (x[1] - x[0]) > 0.1, video_info['segments']))
    if len(segments) == 0:
        continue
    for ann in segments:
        duration_segments.append(ann[1] - ann[0])
    segments_union = IntervalTree.from_tuples(segments)
    segments_union.merge_overlaps()
    actions_duration = sum([i.end - i.begin for i in segments_union])
    percentage_actions.append(actions_duration / video_info['duration'])
    # Check if there is action duration <= 5% of the video duration
    if actions_duration / video_info['duration'] <= 0.05:
        print(video_name, actions_duration / video_info['duration'])

# Print the results (in a table)
from prettytable import PrettyTable
table = PrettyTable()
table.field_names = ["", "min", "max", "mean", "median"]
table.add_row(["Video dur.", f"{min(video_duration):.2f}", f"{max(video_duration):.2f}", f"{sum(video_duration) / len(video_duration):.2f}", f"{sorted(video_duration)[len(video_duration) // 2]:.2f}"])
table.add_row(["Seg. num.", f"{min(num_segments):d}", f"{max(num_segments):d}", f"{sum(num_segments) / len(num_segments):.2f}", f"{sorted(num_segments)[len(num_segments) // 2]:d}"])
table.add_row(["Seg. dur.", f"{min(duration_segments):.2f}", f"{max(duration_segments):.2f}", f"{sum(duration_segments) / len(duration_segments):.2f}", f"{sorted(duration_segments)[len(duration_segments) // 2]:.2f}"])
# table add percentage actions in %
table.add_row(["Action pct.", f"{min(percentage_actions) * 100:.2f}%", f"{max(percentage_actions) * 100:.2f}%", f"{sum(percentage_actions) / len(percentage_actions) * 100:.2f}%", f"{sorted(percentage_actions)[len(percentage_actions) // 2] * 100:.2f}%"])
print(table)

# Plot the boxplot
fig, ax = plt.subplots(1, 4, figsize=(12, 4))
ax[0].boxplot(video_duration)
ax[0].set_title("video duration")
ax[1].boxplot(num_segments)
ax[1].set_title("num segments")
ax[2].boxplot(duration_segments)
ax[2].set_title("duration segments")
ax[3].boxplot(percentage_actions)
ax[3].set_title("percentage actions")
plt.show()

# Print the raw latex codes for showing the table
print(table.get_latex_string())
