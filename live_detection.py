from src.detector import ObjectDetector

print("=" * 45)
print("INTELLIGENT OBJECT DETECTION SYSTEM v1.0")
print("=" * 45)

detector = ObjectDetector()

detector.detect_webcam()

print("\nWEBCAM SESSION ENDED.")