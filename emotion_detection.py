import os
import time
import pickle
import argparse

import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

EMOTIONS = ["angry", "disgust", "happy", "neutral", "sad", "fear", "surprise"]

parser = argparse.ArgumentParser(description="Train, test and save a Random Forest model for the use of detecting a certain emotion in one's face")

parser.add_argument('--estimators', '-e', type=int, default=100, help='The amount of estimators in the Random Forest model')

ESTIMATORS = parser.parse_args().estimators

def format_time(seconds):
    if seconds < 60:
        return f"{seconds} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds %= 60
        return f"{minutes} minutes and {seconds} seconds"
    else:
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        return f"{hours} hours, {minutes} minutes, and {seconds} seconds"

def emotion_data(dataset):

    images = []
    image_emotions = []
    for emotion in tqdm(EMOTIONS, desc=f'Extracting {dataset}ing data'):
        path = os.path.join(os.path.dirname(__file__), "faces", dataset, emotion)
        for file in os.listdir(path):
            with Image.open(os.path.join(path, file)) as image:
                images.append(np.array(image.convert("RGB")))
            image_emotions.append(EMOTIONS.index(emotion))

    model_id = 'openai/clip-vit-base-patch16'
    with torch.inference_mode():
        encoder = AutoModel.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        img_batch = np.array(images)
        img_batch = torch.from_numpy(img_batch)
        img_vecs = encoder(**processor(images=img_batch, return_tensors='np')).image_embeds
    return img_vecs, np.array(image_emotions)

def display_data(predictions, targets, labels, title, losses=False,):
    cm = confusion_matrix(targets, predictions)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=labels)
    _, ax = plt.subplots()
    cm_display.plot(ax=ax)
    ax.set_title(title)
    if losses:
        plt.plot(np.arange(len(losses)), losses)

def train(inputs, targets):
    # Training the model
    return RandomForestClassifier(
        random_state=0,
        verbose=2,
        n_estimators=ESTIMATORS
    ).fit(inputs, targets)

def test(classifier, dataset):
    # Testing the model
    inputs, actuals = emotion_data(dataset)
    results = classifier.predict(inputs)

    correct = 0
    for result, actual in zip(results, actuals):
        if result == actual:
            correct += 1

    display_data(predictions=results, targets=actuals, labels=EMOTIONS, title=f"{dataset.upper()} Data")
    print(f'The model was tested and returned with {round(correct/len(results), 3) * 100}% accuracy')

def save_classifier(classifier):
    if not os.path.isdir("models"):
        os.mkdir("models")
    with open(f"./models/random_forest_{ESTIMATORS}.pickle", 'wb') as file:
        pickle.dump(classifier, file)

if __name__ == '__main__':

    # Start timer
    start = time.perf_counter()

    # Train the model on training data
    inputs, targets = emotion_data("train")
    print(inputs.shape, targets.shape)
    emotion_ai = train(inputs, targets)

    # Test the model on training data
    test(emotion_ai, "train")

    # Test the model on testing data
    test(emotion_ai, "test")

    # Display the results of tests
    plt.show()

    # Save the model
    save_classifier(emotion_ai)

    # Stop timer and calculate elapsed time
    print(f'Emotion model trained, tested and saved in {format_time(round(time.perf_counter() - start))}')
