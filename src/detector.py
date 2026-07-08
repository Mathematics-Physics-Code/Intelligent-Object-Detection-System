from datetime import datetime
import time
import os
from src.analytics import export_session_report
from ultralytics import YOLO

import cv2
import time

from src.config import (
    MODEL_NAME,
    CONFIDENCE_THRESHOLD
)

from src.video_processor import create_video_writer


class ObjectDetector:

    def __init__(self):
        
        self.recording = False
        self.video_writer = None

        os.makedirs("recordings", exist_ok=True)     


        self.session_start_time = None

        self.total_frames = 0

        self.total_fps = 0

        self.session_fps = 0

        print("Initializing Object Detection System...")
        print("Loading YOLO model...")

        self.model = YOLO(MODEL_NAME)
        self.capture_number = 1

        os.makedirs("captures", exist_ok=True)
                
        # Live analytics
        self.current_counts = {}
        self.unique_objects = {}
        self.seen_track_ids = {}


        print("YOLO model loaded successfully!")
        


    # -------------------------------------------------
    # IMAGE DETECTION
    # -------------------------------------------------

    def detect(self, image_path, output_path):

        print("Detecting objects...")

        results = self.model(
            image_path,
            conf=CONFIDENCE_THRESHOLD
        )

        object_counts = {}

        for box in results[0].boxes:

            class_id = int(box.cls[0])

            class_name = self.model.names[class_id]

            confidence = float(box.conf[0])

            print(f"{class_name:<12} {confidence:.2%}")

            if class_name not in object_counts:
                object_counts[class_name] = 1
            else:
                object_counts[class_name] += 1

        print("\nDetection Complete!")

        results[0].save(filename=output_path)

        print(f"Result saved -> {output_path}")

        self.print_statistics(object_counts)

        return object_counts

    # -------------------------------------------------
    # VIDEO DETECTION
    # -------------------------------------------------

    def detect_video(self, video_path, output_path):

        print("Opening video...")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():

            print("Unable to open video.")

            return

        writer = create_video_writer(
            cap,
            output_path
        )

        frame_count = 0

        print("Processing video...\n")

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_count += 1

            results = self.model(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False
            )

            annotated_frame = results[0].plot()

            writer.write(annotated_frame)

            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")

        cap.release()

        writer.release()

        print("\nVideo processing completed!")

        print(f"Saved -> {output_path}")

    # -------------------------------------------------
    # PRINT STATISTICS
    # -------------------------------------------------

    def print_statistics(self, object_counts):

        print("\nDetected Objects")
        print("-" * 25)

        for name, count in object_counts.items():

            print(f"{name:<12}: {count}")

        print(f"\nTotal Objects: {sum(object_counts.values())}")
    # -------------------------------------------------
    # Live Webcam Detection
    # -------------------------------------------------

    def detect_webcam(self):

       from src.logger import logger

       logger.info("Opening webcam.")

       cap = cv2.VideoCapture(0)

       if not cap.isOpened():
          print("Unable to access webcam.")
          return

       print("Press Q to quit.\n")
       self.total_frames = 0
       self.total_fps = 0
       self.current_counts = {}
       self.unique_objects = {}
       self.seen_track_ids = {}
       self.session_start_time = time.time()

       previous_time = time.time()
       while True:

          success, frame = cap.read()
          self.total_frames += 1

          if not success:
            break
          # Reset current frame counts
          self.current_counts = {}
        # -----------------------------
        # Track Objects
        # -----------------------------
          results = self.model.track(
             frame,
             conf=CONFIDENCE_THRESHOLD,
             persist=True,
             tracker="bytetrack.yaml",
             verbose=False
          )
          boxes = results[0].boxes

          if boxes.id is not None:

            for box, track_id in zip(boxes, boxes.id):

                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                track_id = int(track_id)
                                # Current frame count
                self.current_counts[class_name] = (
                    self.current_counts.get(class_name, 0) + 1
                )

                # Unique object count
                if class_name not in self.seen_track_ids:
                    self.seen_track_ids[class_name] = set()

                if track_id not in self.seen_track_ids[class_name]:

                    self.seen_track_ids[class_name].add(track_id)

                    self.unique_objects[class_name] = (
                        self.unique_objects.get(class_name, 0) + 1
                    )
        # -----------------------------
        # Draw Detection Results
        # -----------------------------
          annotated_frame = results[0].plot()

          object_count = len(boxes)

          # -----------------------------
          # FPS
          # -----------------------------
          current_time = time.time()
  
          fps = 1 / (current_time - previous_time)

          previous_time = current_time

          cv2.putText(
            annotated_frame,
            f"FPS: {int(fps)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
          )

          cv2.putText(
            annotated_frame,
            f"Objects: {object_count}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
          )
          self.total_fps += fps
          
          

        # -----------------------------
        # Live Analytics
        # -----------------------------
          y = 120

          cv2.putText(
            annotated_frame,
            "Current Objects",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
          )

          y += 30

          for name, count in self.current_counts.items():

            cv2.putText(
                annotated_frame,
                f"{name}: {count}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            y += 25

          y += 20

          cv2.putText(
            annotated_frame,
            "Unique Objects",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
          )

          y += 30

          for name, count in self.unique_objects.items():

            cv2.putText(
                annotated_frame,
                f"{name}: {count}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            y += 25

        # -----------------------------
        # Display Frame
        # -----------------------------
          if self.recording:

             self.video_writer.write(annotated_frame)
          cv2.imshow(
            "Intelligent Object Detection System v1.0",
            annotated_frame
          )

          key = cv2.waitKey(1) & 0xFF

          if key == ord("s"):

             filename = (
               f"captures/capture_{self.capture_number:03}.jpg"
             )

             cv2.imwrite(filename, annotated_frame)

             logger.info(f"Screenshot saved: {filename}")

             self.capture_number += 1

          elif key == ord("q"):

             break
          elif key == ord("r"):

             if not self.recording:

                filename = datetime.now().strftime(
                   "recordings/session_%Y%m%d_%H%M%S.mp4"
                )

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                self.video_writer = cv2.VideoWriter(
                   filename,
                   fourcc,
                   30,
                   (
                       annotated_frame.shape[1],
                       annotated_frame.shape[0]
                   )
                )

                self.recording = True

                logger.info("Recording started.")

             else:

               self.recording = False

               self.video_writer.release()

               self.video_writer = None

               logger.info("Recording stopped.")
               

       print("\nUnique Objects")
       print("-" * 45)

       for name, count in self.unique_objects.items():
          print(f"{name:<15}{count}")
       cap.release()

       if self.recording:
             self.video_writer.release()

       cv2.destroyAllWindows()
       average_fps = self.total_fps / self.total_frames
       logger.info("Webcam closed.")
       average_fps = self.total_fps / self.total_frames

       session_duration = time.time() - self.session_start_time

       minutes = int(session_duration // 60)
       seconds = int(session_duration % 60)

       print("\n" + "=" * 45)
       print("         SESSION SUMMARY")
       print("=" * 45)

       print(f"Duration        : {minutes} min {seconds} sec")
       print(f"Frames Processed: {self.total_frames}")
       print(f"Average FPS     : {average_fps:.2f}")


       print("=" * 45)
       export_session_report(
          self.unique_objects,
          session_duration,
          self.total_frames,
          average_fps
       )
