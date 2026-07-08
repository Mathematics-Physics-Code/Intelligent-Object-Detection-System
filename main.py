import os

from src.detector import ObjectDetector
from src.image_loader import load_images
from src.result_saver import get_output_path
from src.report_generator import save_report

from src.config import IMAGE_FOLDER
from src.config import OUTPUT_FOLDER


print("=" * 45)
print("INTELLIGENT OBJECT DETECTION SYSTEM")
print("=" * 45)

detector = ObjectDetector()

images = load_images(IMAGE_FOLDER)

print(f"\nFound {len(images)} image(s).\n")

for index, image_path in enumerate(images, start=1):

    print(f"[{index}/{len(images)}] Processing {image_path}")

    output_path = get_output_path(
        OUTPUT_FOLDER,
        image_path
    )

    image_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    report_path = f"logs/{image_name}_report.txt"

    object_counts = detector.detect(
        image_path,
        output_path
    )

    save_report(
        report_path,
        image_name,
        object_counts
    )

    print(f"Report saved -> {report_path}")
    print(f"Image saved  -> {output_path}")

    print("=" * 45)

print("\nALL IMAGES PROCESSED SUCCESSFULLY!")