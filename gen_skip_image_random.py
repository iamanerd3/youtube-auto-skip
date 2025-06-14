import cv2  # For image processing.
import numpy as np  # For numerical operations.
import os  # For directory management.
import random  # For random selection.

def generate_images_with_skip_from_backgrounds(
    background_dir="backgrounds",
    output_dir="images_with_skip_colored",
    count=100,
    width=1920,
    height=1080
):
    # Check and prepare output directory.
    os.makedirs(output_dir, exist_ok=True)

    # Collect all background image paths.
    bg_files = [os.path.join(background_dir, f) for f in os.listdir(background_dir)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    if not bg_files:
        print("❌ No background images found in", background_dir)
        return

    for i in range(count):
        # Choose a random background image.
        bg_path = random.choice(bg_files)
        image = cv2.imread(bg_path)

        if image is None:
            continue  # Skip if image is unreadable.

        # Resize background to match target resolution.
        image = cv2.resize(image, (width, height))

        # Choose random coordinates for text.
        x = random.randint(100, width - 400)
        y = random.randint(150, height - 100)

        # Choose text color based on background contrast (for visibility).
        text_color = (255, 255, 255)  # You may enhance this with dynamic contrast logic.

        # Draw the word "Skip".
        cv2.putText(
            image,
            "Skip",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            text_color,
            6,
            cv2.LINE_AA
        )

        # Save the image.
        filename = os.path.join(output_dir, f"skip_colored_{i+1:03}.jpg")
        cv2.imwrite(filename, image)

    print(f"✅ {count} colorful images with 'Skip' saved in '{output_dir}'.")

# Example usage.
if __name__ == "__main__":
    generate_images_with_skip_from_backgrounds()
