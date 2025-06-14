import cv2  # For image processing.
import pytesseract  # For OCR.
import numpy as np  # For array operations.

# Optional: Specify the path to the tesseract executable if it's not in PATH.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # For Windows.

def detect_skip_word(image_path):
    """
    Detects the word 'Skip' in the input image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        bool: True if 'Skip' is detected, False otherwise.
    """
    # Load the image in color.
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found or cannot be opened.")
        return False

    # Convert to grayscale for better OCR performance.
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to remove background noise.
    _, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optional: Resize to enhance OCR accuracy.
    scaled_image = cv2.resize(thresh_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # Run OCR on the processed image.
    detected_text = pytesseract.image_to_string(scaled_image)

    # Debug: Print OCR result.
    print("Detected Text:\n", detected_text)

    # Check if the word "Skip" appears in the recognized text.
    return "Skip" in detected_text

# Example usage.
if __name__ == "__main__":
    image_path = "a.png"  # Replace with the actual path to your image.
    if detect_skip_word(image_path):
        print("The word 'Skip' was detected in the image.")
    else:
        print("The word 'Skip' was NOT detected in the image.")

