import cv2  # For image processing.
import numpy as np  # For numerical operations.
import os  # For directory management.
import random  # For randomization.

def generate_images_without_skip(output_dir="images_without_skip", count=100, width=1920, height=1080):
    # Create directory if not exists.
    os.makedirs(output_dir, exist_ok=True)

    possible_words = ["Next", "Start", "End", "Play", "Pause", "Menu", "Open", "Exit", "Go", "Wait"]

    for i in range(count):
        # Create a black image.
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Select a random word that is NOT "Skip".
        word = random.choice(possible_words)

        # Generate random position.
        x = random.randint(50, width - 300)
        y = random.randint(100, height - 100)

        # Draw the word.
        cv2.putText(image, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6, cv2.LINE_AA)

        # Save image.
        filename = os.path.join(output_dir, f"noskip_{i+1:03}.jpg")
        cv2.imwrite(filename, image)

    print(f"{count} images without 'Skip' generated in '{output_dir}'.")

# Run the function.
if __name__ == "__main__":
    generate_images_without_skip()
