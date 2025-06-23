import os
import sys
import time

# Set OpenCV log level to FATAL to suppress all but fatal errors
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["FLAGS_log_level"] = "3"  # Suppress PaddleOCR logs

from time import sleep
import cv2
from paddleocr import PaddleOCR

# Initialize PaddleOCR (English) with new parameter
ocr = PaddleOCR(use_textline_orientation=True, lang="en")


# Suppress OpenCV error logs during camera probing, including C-level stderr (Windows)
def suppress_stderr():
    devnull = open(os.devnull, "w")
    devnull_fd = devnull.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(stderr_fd)
    os.dup2(devnull_fd, stderr_fd)
    return devnull, saved_stderr_fd


def restore_stderr(devnull, saved_stderr_fd):
    stderr_fd = sys.stderr.fileno()
    os.dup2(saved_stderr_fd, stderr_fd)
    os.close(saved_stderr_fd)
    devnull.close()


# Only suppress during camera probing

devnull, saved_stderr_fd = (
    suppress_stderr()
)  # Redirect C-level stderr to suppress OpenCV camera errors
try:
    print("[INFO] Probing available camera sources (0-9)...")
    for idx in range(10):
        cap_test = cv2.VideoCapture(idx)
        if cap_test.isOpened():
            print(f"Source {idx} is available.")
            cap_test.release()
        else:
            print(f"Source {idx} is not available.")
finally:
    restore_stderr(devnull, saved_stderr_fd)  # Restore stderr after probing

# Open the selected camera source
cap = cv2.VideoCapture(int(input("Enter the video capture source index (0-9): ")))
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

last_ocr_time = 0  # Timestamp of last OCR run
ocr_interval = 7  # Minimum seconds between OCR runs (controls CPU usage)
last_text = ""  # Last recognized text to display between OCR runs

print("[INFO] Displaying camera feed. OCR will run every", ocr_interval, "seconds.")
try:
    while True:
        ret, frame = cap.read()  # Read a frame from the camera
        if not ret:
            print("Error: Failed to capture frame.")
            break

        now = time.time()  # Current time in seconds
        if now - last_ocr_time > ocr_interval:
            print("[INFO] Running OCR on current frame...")
            # Run OCR on the frame (convert BGR to RGB)
            result = ocr.predict(frame[..., ::-1])
            # Print the result structure for debugging (only once)
            if "printed_predict_debug" not in globals():
                print("ocr.predict() result:", result)
                global printed_predict_debug
                printed_predict_debug = True
            # Extract recognized text lines from rec_texts
            text_lines = [t for t in result[0]["rec_texts"] if t.strip()]
            last_text = " ".join(text_lines)  # Update last recognized text
            print(f"[OCR RESULT] {last_text}")
            last_ocr_time = now  # Update last OCR run time
            print(
                f"[INFO] Returning to camera feed until next OCR in {ocr_interval} seconds."
            )
        text = last_text  # Use last recognized text for display

        # Show the frame with detected text
        display_frame = frame.copy()
        cv2.putText(
            display_frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.imshow("PaddleOCR Camera", display_frame)

        # Check for 'skip' (case-insensitive, anywhere in text)
        if "skip" in text.lower():
            print("SKIP DETECTED, SENDING REQUEST")
            sleep(3)

        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            print("[INFO] 'q' press detected on window, quitting.")
            break
        # If window is closed, cv2.getWindowProperty returns < 0
        if cv2.getWindowProperty("PaddleOCR Camera", cv2.WND_PROP_VISIBLE) < 1:
            print("[INFO] Window close detected, quitting.")
            break
except KeyboardInterrupt:
    print("[INFO] Keyboard interrupt detected, quitting.")
finally:
    cap.release()
    cv2.destroyAllWindows()
