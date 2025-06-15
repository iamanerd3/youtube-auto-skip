from time import sleep
import cv2
from paddleocr import PaddleOCR

# Initialize PaddleOCR (English)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

cap = cv2.VideoCapture(4)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Run OCR on the frame (convert BGR to RGB)
    result = ocr.ocr(frame[..., ::-1], cls=True)
    text_lines = []
    for line in result[0]:
        text_lines.append(line[1][0])

    # Combine detected text lines
    text = " ".join(text_lines)


    # Show the frame with detected text
    display_frame = frame.copy()
    cv2.putText(display_frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('PaddleOCR Camera', display_frame)
    
    # Check for 'skip' (case-insensitive, anywhere in text)
    if "skip" in text.lower():
        print("SKIP DETECTED, SENDING REQUEST")
        sleep(3)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()