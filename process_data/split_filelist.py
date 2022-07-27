# %% Each time only process a subset of a large folder but do NOT want to split the target folder exactly,
# so create a sort of text files, each of which contains a subset of file/sub-folder names of the target folder.
# and these text files can be input as an argument of the relevant function so the function only process a subset.

from pathlib import Path

# Init target folder
t = Path("/home/louis/PycharmProjects/APN/my_data/thumos14/rawframes/train")

# Keep the sub-folder names only. (Optional)
l = [i.name for i in t.iterdir()]

# Split into 100-length chunks
k = [l[i:i + 100] for i in range(0, len(l), 100)]

# Write into files

for i, d in enumerate(k):
    with open(f"train_split_{i}.txt", 'w') as f:
        f.write("\n".join(str(item) for item in d))

# Run command
import subprocess

subprocess.run(
    "bash dist_extract.sh configs/extract_gma.py checkpoints/gma_8x2_120k_mixed_368x768.pth 2 --src-dir my_data/thumos14/rawframes/train --out-dir my_data/thumos14/gma_flow/train --video-names my_data/thumos14/annotations/splits/train-100/train_split_1.txt",
    shell=True)
