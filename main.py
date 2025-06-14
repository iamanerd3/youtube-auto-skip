import cv2
import pytesseract
from PIL import Image

# Path to the tesseract executable (if not in your system's PATH)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Replace with your tesseract path

# Initialize camera
cap = cv2.VideoCapture(0)  # 0 is usually the default camera, use 1, 2, ... for others

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    frame_count += 1
    print(f"Frame {frame_count}: ret={ret}, shape={frame.shape if ret else 'N/A'}")

    if not ret:
        print("Error: Failed to capture frame. Exiting loop.")
        break

    # Show the raw camera frame for debugging
    cv2.imshow('Raw Camera', frame)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use Pillow to convert the OpenCV frame to an image compatible with pytesseract
    pil_image = Image.fromarray(gray)

    # Perform OCR
    text = pytesseract.image_to_string(pil_image)
    print(f"OCR Text: {repr(text)}")

    # Display the frame and the extracted text
    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Live OCR', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()