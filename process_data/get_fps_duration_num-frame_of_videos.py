from pathlib import Path
import mmengine
import cv2

p1 = Path("/home/louis/PycharmProjects/APN/my_data/thumos14/videos/videos/val")
p2 = Path("/home/louis/PycharmProjects/APN/my_data/thumos14/videos/videos/test")
video_info = {}
fps_dict = {}
dur_dict = {}
fra_dict = {}
for p in [p1, p2]:
    for v in p.iterdir():
        reader = cv2.VideoCapture(str(v))
        fps = reader.get(cv2.CAP_PROP_FPS)
        fra = reader.get(cv2.CAP_PROP_FRAME_COUNT)
        fps_dict[v.stem] = fps
        dur_dict[v.stem] = fra / fps
        fra_dict[v.stem] = int(fra)
        video_info[v.stem] = {'fps': fps, 'dur': fra / fps, 'fra': int(fra)}

# # Validate if the number of extracted frames equals the cv2.CAP_PROP_FRAME_COUNT. Conclusion: Yes, all are aligned.
# fra_2_dict = {}
# for folder in Path("/mnt/louis-consistent/Datasets/thumos14/rawframes/val").iterdir():
#     num_frame = len(list(folder.glob('img*')))
#     fra_2_dict[folder.name] = num_frame
# xx = []
# for k, v in fra_2_dict.items():
#     print(type(fra_dict[k]), type(v))
#     if fra_dict[k] != v:
#         xx.append({'fra_1': fra_dict[k], 'fra_2': v})
# print(xx)   # empty

# # Validate if the real FPS equals the cv2.CAP_PROP_FPS. Conclusion: Yes, all aligned
# fps_25 = [k for k, v in fps_dict.items() if v == 25]
# dur_25 = [dur_dict[k] for k in fps_25]
# # Example: duration computed by num_frames/fps of video_validation_0000311 is 17475 / 25 = 699 seconds.
# # Open this video in video player it showed that duration is 11:40 = 700 seconds.
# fps_30 = [k for k, v in fps_dict.items() if v == 30]
# dur_30 = [dur_dict[k] for k in fps_30]
# # Example: duration computed by num_frames/fps of video_validation_0000170 is 6476 / 30 = 215.9 seconds.
# # Open this video in video player it showed that duration is 03:36 = 216 seconds.

print(
    """
    Conclusion: The cv2.CAP_PROP_FRAME_COUNT and cv2.CAP_PROP_FPS all correctly give the correct property of videos.
    Concretely, the cv2.CAP_PROP_FRAME_COUNT align with the number of extracted frames (by denseflow which uses ffmpeg)
    the cv2.CAP_PROP_FPS align with the video player (Ubuntu, by checking videos' duration).
    """
)
mmengine.dump(video_info, "/home/louis/PycharmProjects/APN/my_data/thumos14/th14_video_info.pkl")
