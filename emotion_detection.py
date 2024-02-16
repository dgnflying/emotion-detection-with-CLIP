from sklearn.neural_network import MLPClassifier
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

emotions = ["angry", "disgusted", "happy", "neutral", "sad", "scared", "surprised"]

def emotion_data(set):
    images = []
    image_emotions = []
    for emotion in tqdm(emotions, desc='Extracting training data'):
        path = os.path.join(os.path.dirname(__file__), "faces", set, emotion)
        for file in os.listdir(path):
            with Image.open(os.path.join(path, file)) as image:
                images.append(np.array(image).flatten())
            image_emotions.append(emotions.index(emotion))
    return np.array(images), np.array(image_emotions)

def display_test_data(predictions, targets, losses, labels, title):
    plt.plot(np.arange(len(losses)), losses)
    cm = confusion_matrix(targets, predictions)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=labels)
    _, ax = plt.subplots()
    cm_display.plot(ax=ax)
    ax.set_title(title)
    plt.show()

def train(inputs, targets):
    # Training the model
    classifier = MLPClassifier(
        random_state=0,
        verbose=1,
        hidden_layer_sizes=(3000, 500, 200, 100),
    )
    classifier.fit(inputs, targets)

    return classifier

def test(classifier, test, actual):
    # Testing the model
    results = classifier.predict(test)

    correct = 0
    for i in range(len(results)):
        if results[i] == actual[i]:
            correct += 1

    print(f'The model was tested: {round(correct/len(results), 3) * 100}% accuracy')

    display_test_data(results, actual, classifier.loss_curve_, emotions, "Train Data")

if __name__ == '__main__':
    inputs, targets = emotion_data("train")
    emotion_ai = train(inputs, targets)
    test_inputs, test_actuals = emotion_data("test")
    test(emotion_ai, test_inputs, test_actuals)