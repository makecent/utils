import argparse
import mmengine
from pathlib import Path

import cv2


def main(data_root, output):
    data_root = Path(data_root)
    output = Path(output)
    output.mkdir(exist_ok=True, parents=True)
    video_info = {}
    fps_dict = {}
    dur_dict = {}
    fra_dict = {}
    for video in mmengine.track_iter_progress(list(data_root.iterdir())):
        reader = cv2.VideoCapture(str(video))
        fps = reader.get(cv2.CAP_PROP_FPS)
        fra = reader.get(cv2.CAP_PROP_FRAME_COUNT)
        # get video shape, H, W
        h, w = reader.get(cv2.CAP_PROP_FRAME_HEIGHT), reader.get(cv2.CAP_PROP_FRAME_WIDTH)
        fps_dict[video.stem] = fps
        dur_dict[video.stem] = fra / fps
        fra_dict[video.stem] = int(fra)
        video_info[video.stem] = {'fps': fps,
                                  'duration': fra / fps,
                                  'num_frame': int(fra),
                                  'height': int(h),
                                  'width': int(w)}

    mmengine.dump(video_info, output / 'video_info.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, help='path to the raw videos')
    parser.add_argument('--output', type=str, default='.', help='path to save the video info')
    args = parser.parse_args()
    main(args.data_root, args.output)

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
