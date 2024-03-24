import json

# Assuming thumos14_test.json is also loaded correctly

# Load both annotation files
with open('assets/tad_annotations/TH14/thumos14_val.json', 'r') as file:
    val_annotations = json.load(file)

with open('assets/tad_annotations/TH14/thumos14_test.json', 'r') as file:
    test_annotations = json.load(file)

# Merge the dictionaries from both files
annotations = {**val_annotations, **test_annotations}

# Re-define bins including 0.5%
bins = [0.5, 1, 3, 4, 6, 10, 16, 25, 40, 63, 100]

# Initialize the counts for each bin
bin_counts = {bin: 0 for bin in bins}
total_instances = 0

# Process annotations to calculate and bin percentage durations
for video in annotations.values():
    video_duration = video['duration']
    for segment in video['segments']:
        instance_duration = segment[1] - segment[0]
        percentage_duration = (instance_duration / video_duration) * 100

        for bin in bins:
            if percentage_duration <= bin:
                bin_counts[bin] += 1
                break

        total_instances += 1

# Calculate percentages for THUMOS dataset
thumos_percentages = {bin: (count / total_instances) * 100 for bin, count in bin_counts.items()}

# Provided COCO dataset statistics for comparison
coco_statistics = {4: 29, 6: 36, 10: 10, 16: 8, 25: 7, 40: 5, 63: 4, 100: 1}

# Plotting
import matplotlib.pyplot as plt

# Convert bin percentages for plotting
bins_percentage = list(thumos_percentages.keys())
thumos_percentage_values = list(thumos_percentages.values())
coco_bins_percentage = list(coco_statistics.keys())
coco_percentage_values = [coco_statistics[bin] for bin in coco_bins_percentage]

# Combined bins from both datasets, ensuring all unique bins are represented
combined_bins = sorted(set(bins + coco_bins_percentage))

# Mapping each bin to its index for equal spacing
bins_index_map = {bin: i for i, bin in enumerate(combined_bins)}

# Preparing THUMOS data for plotting
thumos_plot_x = [bins_index_map[bin] for bin in bins_percentage]
thumos_plot_y = thumos_percentage_values

# Preparing COCO data for plotting, filtering out bins not in coco_statistics to avoid plotting 0% points
coco_plot_x = [bins_index_map[bin] for bin in coco_bins_percentage if bin in coco_statistics]
coco_plot_y = [coco_statistics[bin] for bin in coco_bins_percentage if bin in coco_statistics]

# Plotting
plt.figure(figsize=(6, 5))
plt.plot(thumos_plot_x, thumos_plot_y, marker='o', linestyle='-', color='#4A7298', label='THUMOS', markersize=8)
plt.plot(coco_plot_x, coco_plot_y, marker='s', linestyle='-', color='#F3C846', label='COCO', markersize=8)

# Adjust the plot with equal interval x-ticks
plt.xticks(range(len(combined_bins)), [f"{bin}%" for bin in combined_bins], rotation=45)
plt.yticks(range(0, 41, 5), [f"{y}%" for y in range(0, 41, 5)])
plt.title('Distribution of Instance Size (THUMOS vs COCO)')
plt.xlabel('Percent of Space Occupied')
plt.ylabel('Percent of Instances')
plt.grid(True)
plt.ylim(0, 40)
plt.legend()

plt.tight_layout()
plt.show()