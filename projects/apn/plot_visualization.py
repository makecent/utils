import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import mmengine

# Video annotations file
video_annotations = pd.read_csv(r"projects\apn\src\apn_th14_test.csv", header=None)

# Detection results for 210 videos. A dictionary with class index as key and a list of detection results as value.
"""11: array([['video_test_0000188', '11', '15093.44', '15319.68',
         '0.9891499565998264'],
        ['video_test_0000188', '11', '662.5600000000001', '824.16',
         '0.975999903999616'],
        ['video_test_0000188', '11', '14511.68', '14689.44',
         '0.966368047290371'],
        ...,
        ['video_test_0001098', '11', '1540.674', '2752.272',
         '0.29742963338715667'],
        ['video_test_0001098', '11', '2423.196', '4931.1539999999995',
         '0.22558263555027136'],
        ['video_test_0001098', '11', '4417.596', '4796.532',
         '0.017021050996028908']], dtype='<U32'),"""
detection_results = mmengine.load("projects/apn/src/th14_detections.pkl")

# Predicted progression sequence for 210 videos. 210000, each video has 1000 frames sampled for prediction.
all_progs = mmengine.load(r"projects\apn\src\rgb+flow_progressions.pkl")

# The video to be plot
video_name = "video_test_0000062.mp4"
video_index = int()  # to find the index of the video in the detection results
video_prog = all_progs[video_index*1000:video_index*1000+1000]
# we plot the first ground truth in the video by default
ground_truth = video_annotations[video_annotations[0] == video_name].iloc[0]

# the range of plot is the triple of the ground truth
len_ground_truth = int(ground_truth[3]) - int(ground_truth[2])
min_frame = int(ground_truth[2]) - len_ground_truth
max_frame = int(ground_truth[3]) + len_ground_truth

# Simulate frame index and progression based on detection results
frame_index = np.linspace(min_frame, max_frame, num=100)  # Generating 100 sample frame indices
predicted_progression = np.interp(frame_index, [min_frame, max_frame], [0, 100])  # Linear interpolation for progression

fig = plt.figure(figsize=(15, 10))

# First Row: Simulate "Images" with colored rectangles as placeholders
for i in range(1, 6):
    ax = fig.add_subplot(3, 5, i)
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, color=np.random.rand(3,)))  # Random colors for placeholder "images"
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

# Second Row: Plot y-x curve and rectangle for predicted boundaries
# Using the detection results to define the rectangle for predicted boundaries
predicted_start, predicted_end = detection_results[0, 2], detection_results[-1, 3]  # Assuming detection spans all results

ax2 = fig.add_subplot(3, 1, 2)
ax2.plot(frame_index, predicted_progression)
ax2.add_patch(patches.Rectangle((predicted_start, 0), predicted_end-predicted_start, 100, color='orange', alpha=0.3))
ax2.set_ylim(0, 100)
ax2.set_ylabel('Action Progression (%)')
ax2.set_xlabel('Frame Index')

# Third Row: Empty rectangle with sub-rectangles for ground truth boundary
ax3 = fig.add_subplot(3, 1, 3)
ax3.add_patch(patches.Rectangle((min_frame, 0), max_frame-min_frame, 1, color='white', edgecolor='black'))

# Adding red sub-rectangles for each ground truth annotation
for index, row in video_annotations.iterrows():
    ax3.add_patch(patches.Rectangle((row['start_frame'], 0), row['end_frame']-row['start_frame'], 1, color='red'))
ax3.set_xlim(min_frame, max_frame)
ax3.axis('off')

plt.tight_layout()
plt.show()

