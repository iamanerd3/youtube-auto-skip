from imageai.Classification.Custom import ClassificationModelTrainer

model_trainer = ClassificationModelTrainer()
model_trainer.setModelTypeAsDenseNet121()
model_trainer.setDataDirectory(r".")
model_trainer.trainModel(num_experiments=20, verbose=True)
