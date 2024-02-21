from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import time
import pickle

emotions = ["angry", "disgusted", "happy", "neutral", "sad", "scared", "surprised"]

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
        n_estimators=10000
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
    with open("./model/random_forest.pickle", 'wb') as file:
        pickle.dump(classifier, file)


if __name__ == '__main__':
    start = time.perf_counter()
    inputs, targets = emotion_data("train")
    emotion_ai = train(inputs, targets)
    test(emotion_ai, "train")
    test(emotion_ai, "test")
    plt.show()
    save_classifier(emotion_ai)
    print()
    elapsed = (time.perf_counter() - start) // 60
    if elapsed > 60:
        elapsed //= 60
        elapsed = f"{elapsed} hours"
    else:
        elapsed = f"{elapsed} minutes"

    print(f'Emotion model trained, tested and saved in {elapsed} minutes')