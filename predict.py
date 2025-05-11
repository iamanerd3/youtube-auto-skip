from imageai.Classification.Custom import CustomImageClassification
import os
from cv2 import VideoCapture, imwrite

# Initialize the camera
cam = VideoCapture(0)  # 0 -> index of camera
execution_path = os.getcwd()
models_path = os.path.join(execution_path, "models")

# List all .pt files in the ./models directory
model_files = [f for f in os.listdir(models_path) if f.endswith(".pt")]

if not model_files:
    print("No .pt files found in the ./models directory.")
    exit()

# Display the list of models and ask the user to select one
print("Available models:")
for idx, model_file in enumerate(model_files):
    print(f"{idx}: {model_file}")

selected_index = int(
    input("Enter the number corresponding to the model you want to use: ")
)

if selected_index < 0 or selected_index >= len(model_files):
    print("Invalid selection. Exiting.")
    exit()

selected_model = model_files[selected_index]
print(f"Selected model: {selected_model}")

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


def captureCameraImage():
    s, img = cam.read()
    if s:  # Frame captured without any errors
        imwrite("camera.jpg", img)  # Save image


def predict(image_path: str):
    classify = prediction.classifyImage(
        os.path.join(
            execution_path,
            image_path,
        ),
        result_count=1,
    )

    predictions, probabilities = classify[0], classify[1]

    for eachPrediction, eachProbability in zip(predictions, probabilities):
        print(eachPrediction, " : ", eachProbability)


if __name__ == "__main__":
    while True:
        captureCameraImage()
        predict("camera.jpg")
