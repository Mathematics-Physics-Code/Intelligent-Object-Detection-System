import csv
import os
from datetime import datetime


def export_session_report(
    unique_objects,
    session_duration,
    total_frames,
    average_fps
):

    os.makedirs("analytics", exist_ok=True)

    filename = datetime.now().strftime(
        "analytics/session_%Y%m%d_%H%M%S.csv"
    )

    with open(filename, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow(["Metric", "Value"])
        writer.writerow(["Duration (seconds)", round(session_duration, 2)])
        writer.writerow(["Frames Processed", total_frames])
        writer.writerow(["Average FPS", round(average_fps, 2)])
        writer.writerow([])
        writer.writerow(["Object", "Unique Count"])

        for name, count in unique_objects.items():
            writer.writerow([name, count])

    print(f"\nAnalytics saved -> {filename}")