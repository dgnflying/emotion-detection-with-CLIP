from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import time

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

def display_test_data(predictions, targets, labels, title, losses=False,):
    cm = confusion_matrix(targets, predictions)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=labels)
    _, ax = plt.subplots()
    cm_display.plot(ax=ax)
    ax.set_title(title)
    if losses:
        plt.plot(np.arange(len(losses)), losses)
    plt.show()

def train(inputs, targets):
    # classifier = MLPClassifier(
    #     random_state=0,
    #     verbose=1,
    #     hidden_layer_sizes=(1000, 100, 50),
    # )
    # Training the model
    classifier = RandomForestClassifier(
        random_state=0,
        verbose=2,
        n_estimators=5000
    )
    classifier.fit(inputs, targets)
    classifier

    return classifier

def test(classifier, tests, actuals):
    # Testing the model
    results = classifier.predict(tests)

    correct = 0
    for result, actual in zip(results, actuals):
        if result == actual:
            correct += 1

    display_test_data(predictions=results, targets=actuals, labels=emotions, title="Train Data")

    print(f'The model was tested and returned with {round(correct/len(results), 3) * 100}% accuracy')
if __name__ == '__main__':
    start = time.perf_counter()
    inputs, targets = emotion_data("train")
    emotion_ai = train(inputs, targets)
    test_inputs, test_actuals = emotion_data("test")
    test(emotion_ai, test_inputs, test_actuals)
    print()
    print(f'Emotion model trained and tested in {(time.perf_counter() - start) // 60} minutes')