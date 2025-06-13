from imageai.Classification.Custom import ClassificationModelTrainer

model_trainer = ClassificationModelTrainer()
models_with_selectors = {
    "ResNet50": model_trainer.setModelTypeAsResNet50,
    "InceptionV3": model_trainer.setModelTypeAsInceptionV3,
    "DenseNet121": model_trainer.setModelTypeAsDenseNet121,
    "MobileNetV2": model_trainer.setModelTypeAsMobileNetV2,
}


def get_model_choices():
    print("Select models to train (comma separated numbers):")
    for index, model in enumerate(models_with_selectors.keys(), 1):
        print(f"{index}. {model}")

    selected_indexes = input("Enter your choice(s): ")
    selected_indexes = selected_indexes.split(",")  # split by comma
    selected_indexes = [
        int(x.strip()) - 1 for x in selected_indexes
    ]  # convert to zero-indexed

    for i in selected_indexes:
        if i < 0 or i >= len(models_with_selectors):
            raise ValueError(f"Invalid selection: {i + 1}. Exiting.")

    selected_models = [list(models_with_selectors.keys())[i] for i in selected_indexes]

    print(f"Selected models: {', '.join(selected_models)}")
    return selected_models


def box_print(
    text, left_border="|", right_border="|", top_border="-", bottom_border="-"
):
    """
    Print a box around the given text.

    Use with one line of text.

    Parameters
    ----------
    text : str
        The text to print inside the box.
    left_border : str, optional
        The character to use for the left border (default is "|").
    right_border : str, optional
        The character to use for the right border (default is "|").
    top_border : str, optional
        The character to use for the top border (default is "-").
    bottom_border : str, optional
        The character to use for the bottom border (default is "-").

    Returns
    -------
    None
    """

    middle_line_message = f"{left_border} {text} {right_border}"
    length = len(middle_line_message)

    print(top_border * length)
    print(middle_line_message)
    print(bottom_border * length)


selected_model_names = get_model_choices()
model_trainer.setDataDirectory(r".")

for model_name in selected_model_names:
    box_print(f"Selecting model: {model_name}")
    models_with_selectors[model_name]()

    box_print(f"Training model: {model_name}...")
    model_trainer.trainModel(num_experiments=2, verbose=True)
