import cv2
import os
import time
import argparse
import datetime

def take_snapshots(device_path, interval_seconds, output_folder):
    # Ensure output folder exists. Create it if not.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Attempt to open the video capture device.
    video_capture = cv2.VideoCapture(device_path)
    if not video_capture.isOpened():
        print(f"Error: Cannot open video device {device_path}.")
        return

    print(f"Capturing snapshots from {device_path} every {interval_seconds} second(s).")
    print(f"Saving snapshots to {output_folder}. Press Ctrl+C to stop.")

    try:
        while True:
            # Read a frame from the video device.
            success, frame = video_capture.read()
            if not success:
                print("Failed to capture frame. Retrying...")
                time.sleep(interval_seconds)
                continue

            # Create a timestamped filename.
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.jpg"
            filepath = os.path.join(output_folder, filename)

            # Save the image to the output folder.
            cv2.imwrite(filepath, frame)
            print(f"Saved snapshot: {filepath}")

            # Wait for the specified interval before capturing next image.
            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\nCapture stopped by user.")

    finally:
        # Release the video capture device.
        video_capture.release()
        print("Video device released.")

def main():
    # Set up argument parser for command-line input.
    parser = argparse.ArgumentParser(description="Capture snapshots from a video device at fixed intervals.")
    parser.add_argument("device", help="Path to video device (e.g., /dev/video0).")
    parser.add_argument("interval", type=int, help="Time interval between snapshots (in seconds).")
    parser.add_argument("folder", help="Folder to save snapshots.")

    args = parser.parse_args()

    # Start the snapshot process.
    take_snapshots(args.device, args.interval, args.folder)

if __name__ == "__main__":
    main()
