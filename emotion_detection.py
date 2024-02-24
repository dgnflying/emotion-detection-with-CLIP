import os
import time
import pickle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tqdm import tqdm

estimators = 10
emotions = ["angry", "disgust", "happy", "neutral", "sad", "fear", "surprise"]

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


def emotion_data(set):
    images = []
    image_emotions = []
    for emotion in tqdm(emotions, desc=f'Extracting {set}ing data'):
        path = os.path.join(os.path.dirname(__file__), "faces", set, emotion)
        for file in os.listdir(path):
            with Image.open(os.path.join(path, file)) as image:
                images.append(np.array(image).flatten())
            image_emotions.append(emotions.index(emotion))
    return np.array(images), np.array(image_emotions)

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
        n_estimators=estimators
    ).fit(inputs, targets)

def test(classifier, set):
    # Testing the model
    inputs, actuals = emotion_data(set)
    results = classifier.predict(inputs)

    correct = 0
    for result, actual in zip(results, actuals):
        if result == actual:
            correct += 1

    display_data(predictions=results, targets=actuals, labels=emotions, title=f"{set.upper()} Data")
    print(f'The model was tested and returned with {round(correct/len(results), 3) * 100}% accuracy')

def save_classifier(classifier):
    if not os.path.isdir("models"):
        os.mkdir("models")
    with open(f"./models/random_forest_{estimators}.pickle", 'wb') as file:
        pickle.dump(classifier, file)


if __name__ == '__main__':

    # Start timer
    start = time.perf_counter()

    # Train the model on training data
    inputs, targets = emotion_data("train")
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
