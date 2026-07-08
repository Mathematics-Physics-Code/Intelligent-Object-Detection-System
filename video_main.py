from src.config import VIDEO_FOLDER
from src.config import OUTPUT_VIDEO_FOLDER

from src.result_saver import get_output_path

from src.detector import ObjectDetector
from src.video_loader import load_videos



print("=" * 45)
print("VIDEO OBJECT DETECTION")
print("=" * 45)

detector = ObjectDetector()

videos = load_videos(VIDEO_FOLDER)

print(f"\nFound {len(videos)} video(s).\n")

for index, video_path in enumerate(videos, start=1):

    print(f"[{index}/{len(videos)}] Processing {video_path}")

    output_path = get_output_path(
        OUTPUT_VIDEO_FOLDER,
        video_path
    )

    detector.detect_video(
        video_path,
        output_path
    )
print("\nALL VIDEOS PROCESSED SUCCESSFULLY!")