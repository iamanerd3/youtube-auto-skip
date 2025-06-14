import cv2  # For image processing.
import numpy as np  # For numerical operations.
import os  # For directory management.
import random  # For random placement.

def generate_images_with_skip(output_dir="images_with_skip", count=100, width=1920, height=1080):
    # Create directory if not exists.
    os.makedirs(output_dir, exist_ok=True)

    for i in range(count):
        # Create a black image.
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Generate random position for the text.
        x = random.randint(50, width - 300)
        y = random.randint(100, height - 100)

        # Draw the word "Skip" in white.
        cv2.putText(image, "Skip", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6, cv2.LINE_AA)

        # Save image to file.
        filename = os.path.join(output_dir, f"skip_{i+1:03}.jpg")
        cv2.imwrite(filename, image)

    print(f"{count} images with 'Skip' generated in '{output_dir}'.")

# Run the function.
if __name__ == "__main__":
    generate_images_with_skip()
