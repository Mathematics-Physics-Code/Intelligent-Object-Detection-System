import os


def save_report(report_path, image_name, object_counts):

    with open(report_path, "w") as file:

        file.write("=" * 40 + "\n")
        file.write("INTELLIGENT OBJECT DETECTION REPORT\n")
        file.write("=" * 40 + "\n\n")

        file.write(f"Image: {image_name}\n\n")

        file.write("Detected Objects\n")
        file.write("-" * 25 + "\n")

        for name, count in object_counts.items():

            file.write(f"{name:<12}: {count}\n")

        file.write(f"\nTotal Objects: {sum(object_counts.values())}\n")