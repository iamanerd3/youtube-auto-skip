import os
import shutil
import cv2
import numpy as np
import random
import albumentations as A

# Supported image extensions
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

# Set your input and output directories here
INPUT_DIR = "./original"
OUTPUT_DIR = "./train"

GEN_SUFFIX = ""


def add_random_scribbles(image):
    """Add random lines, dots, and scribbles with random color, thickness, transparency, etc."""
    overlay = image.copy()
    h, w = image.shape[:2]
    num_elements = random.randint(1, 3)
    alpha = random.uniform(0.2, 0.7)

    for _ in range(num_elements):
        element_type = random.choice(["line", "dot", "scribble"])
        color = [random.randint(0, 255) for _ in range(3)]
        alpha = random.uniform(0.2, 0.7)
        thickness = random.randint(1, 8)
        if element_type == "line":
            pt1 = (random.randint(0, w - 1), random.randint(0, h - 1))
            pt2 = (random.randint(0, w - 1), random.randint(0, h - 1))
            cv2.line(overlay, pt1, pt2, color, thickness)
        elif element_type == "dot":
            center = (random.randint(0, w - 1), random.randint(0, h - 1))
            radius = random.randint(2, 15)
            cv2.circle(overlay, center, radius, color, -1)
        elif element_type == "scribble":
            num_points = random.randint(3, 8)
            points = np.array(
                [
                    [random.randint(0, w - 1), random.randint(0, h - 1)]
                    for _ in range(num_points)
                ],
                np.int32,
            )
            points = points.reshape((-1, 1, 2))
            cv2.polylines(overlay, [points], False, color, thickness)
    # Blend overlay with original image
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image


def transform_image(image):
    height, width = image.shape[:2]
    # Random crop: crop between 40% and 80% of the original size
    crop_height = random.randint(int(height * 0.4), int(height * 0.8))
    crop_width = random.randint(int(width * 0.4), int(width * 0.8))
    x_min = random.randint(0, width - crop_width)
    y_min = random.randint(0, height - crop_height)
    cropped = image[y_min : y_min + crop_height, x_min : x_min + crop_width]

    # Slightly less extreme blur
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.01),
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.5),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=(5, 13), p=1.0),
                    A.MedianBlur(blur_limit=9, p=1.0),
                    A.Blur(blur_limit=9, p=1.0),
                    A.GaussianBlur(blur_limit=(5, 13), sigma_limit=(1.0, 5.0), p=1.0),
                ],
                p=1.0,
            ),
            A.RandomGamma(gamma_limit=(60, 180), p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=30, sat_shift_limit=50, val_shift_limit=50, p=0.5
            ),
            A.Resize(height, width),  # Resize back to original size
        ]
    )
    augmented = transform(image=cropped)
    noisy = add_random_scribbles(augmented["image"])
    return noisy


def process_folder(folder_path, output_base, gen_suffix=GEN_SUFFIX):
    parent, folder_name = os.path.split(folder_path.rstrip(os.sep))
    output_folder = os.path.join(output_base, folder_name + gen_suffix)
    print(f"\n---------- {folder_name}{gen_suffix} ----------")
    os.makedirs(output_folder, exist_ok=True)

    # Gather all image files first
    image_files = [
        fname for fname in os.listdir(folder_path) if fname.lower().endswith(IMAGE_EXTS)
    ]
    total = len(image_files) * 11  # 1 original + 10 augmentations per file
    count = 0

    for fname in image_files:
        img_path = os.path.join(folder_path, fname)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        name, ext = os.path.splitext(fname)

        # Save the original image as [name]-0.ext
        orig_out_path = os.path.join(output_folder, f"{name}-0{ext}")
        count += 1
        percent = int((count / total) * 100)
        print(f"({percent}%) ----- {name}-0{ext} -----")
        print(f"({percent}%) {os.path.relpath(orig_out_path)}")
        cv2.imwrite(orig_out_path, image)

        # Save 10 augmented versions as [name]-1.ext ... [name]-10.ext
        for i in range(1, 11):
            transformed = transform_image(image_rgb)
            output_image = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)
            out_path = os.path.join(output_folder, f"{name}-{i}{ext}")
            count += 1
            percent = int((count / total) * 100)
            print(f"({percent}%) {os.path.relpath(out_path)}")
            cv2.imwrite(out_path, output_image)


if __name__ == "__main__":
    # Only delete folders if GEN_SUFFIX is not empty
    if GEN_SUFFIX and os.path.exists(OUTPUT_DIR):
        for entry in os.listdir(OUTPUT_DIR):
            full_path = os.path.join(OUTPUT_DIR, entry)
            if os.path.isdir(full_path) and entry.endswith(GEN_SUFFIX):
                shutil.rmtree(full_path)
    else:
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Now process all non-_gen folders from the input directory
    for entry in os.listdir(INPUT_DIR):
        full_path = os.path.join(INPUT_DIR, entry)
        # Only skip folders ending with GEN_SUFFIX if GEN_SUFFIX is not empty
        if os.path.isdir(full_path) and (
            not GEN_SUFFIX or not entry.endswith(GEN_SUFFIX)
        ):
            process_folder(full_path, OUTPUT_DIR, GEN_SUFFIX)
