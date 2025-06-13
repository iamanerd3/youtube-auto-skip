import sys
from imageai.Classification.Custom import CustomImageClassification
import os
from cv2 import VideoCapture, imwrite
import re

# Initialize the camera
# List all available camera sources (indexes 0-9)
print("Available camera sources:")
for idx in range(10):
    cap = VideoCapture(idx)
    if cap.isOpened():
        print(f"Camera index {idx} is available.")
        cap.release()
cameraIndex = int(
    input("Enter camera index (default 0): ") or 0
)  # let user choose camera source
cam = VideoCapture(cameraIndex)  # 0 -> index of camera

execution_path = os.getcwd()
models_path = os.path.join(execution_path, "models")


def capture_camera_image():
    s, img = cam.read()
    if s:  # Frame captured without any errors
        imwrite("camera.jpg", img)  # Save image


def predict(image_path: str):
    classify = prediction.classifyImage(
        os.path.join(
            execution_path,
            image_path,
        ),
        result_count=2,
    )

    predictions, probabilities = classify[0], classify[1]

    delete_multiple_lines(len(predictions))
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        print(eachPrediction, " : ", eachProbability)


def ask_for_model():
    # List all .pt files in the ./models directory
    model_files = [f for f in os.listdir(models_path) if f.endswith(".pt")]

    if not model_files:
        print("No .pt files found in the ./models directory.")
        exit()

    # Function to extract accuracy from the model filename
    def extract_accuracy(model_file):
        match = re.search(r"test_acc_([0-9.]+)", model_file)
        if match:
            return float(match.group(1))
        return 0  # default to 0 if no accuracy found

    # Sort model files by accuracy (highest to lowest)
    model_files.sort(key=extract_accuracy, reverse=True)

    # Display the sorted list of models
    print("Available models sorted by accuracy:")
    for idx, model_file in enumerate(model_files):
        accuracy = extract_accuracy(model_file)
        print(f"{idx}: {model_file} - Accuracy: {accuracy}")

    selected_index = int(
        input("Enter the number corresponding to the model you want to use: ")
    )

    if selected_index < 0 or selected_index >= len(model_files):
        print("Invalid selection. Exiting.")
        exit()

    global selected_model
    selected_model = model_files[selected_index]
    print(f"Selected model: {selected_model}")


def load_model():
    global prediction

    # Determine the model type based on the file name
    prediction = CustomImageClassification()

    if "inception_v3" in selected_model.lower():
        prediction.setModelTypeAsInceptionV3()
    elif "densenet121" in selected_model.lower():
        prediction.setModelTypeAsDenseNet121()
    elif "mobilenet_v2" in selected_model.lower():
        prediction.setModelTypeAsMobileNetV2()
    elif "resnet50" in selected_model.lower():
        prediction.setModelTypeAsResNet50()
    else:
        print("Could not determine model type from the file name. Exiting.")
        exit()

    # Load the selected model
    prediction.setModelPath(os.path.join(models_path, selected_model))
    prediction.setJsonPath(os.path.join(models_path, "._model_classes.json"))
    prediction.loadModel()


def delete_multiple_lines(n=1):
    """Delete the last line in the STDOUT."""
    for _ in range(n):
        sys.stdout.write("\x1b[1A")  # cursor up one line
        sys.stdout.write("\x1b[2K")  # delete the last line


if __name__ == "__main__":
    ask_for_model()
    load_model()

    os.system("cls" if os.name == "nt" else "printf '\033c'")
    try:
        while True:
            capture_camera_image()
            predict("camera.jpg")
    except KeyboardInterrupt:
        print("\nExiting...")
