from sklearn.neural_network import MLPClassifier
from PIL import Image
import numpy as np
import os
import random
from tqdm import tqdm

emotions = ["angry", "disgusted", "happy", "neutral", "sad", "scared", "surprised"]

def learn_emotions():
    # Training the model
    inputs = []
    targets = []
    for emotion in tqdm(emotions, desc='Extracting train data'):
        path = os.path.join(os.path.dirname(__file__), "faces", "train", emotion)
        for file in os.listdir(path):
            with Image.open(os.path.join(path, file)) as image:
                inputs.append(np.array(image).flatten())
            targets.append(emotions.index(emotion))
    inputs = np.array(inputs)
    targets = np.array(targets)

    print(f'Shape of the input data:  {inputs.shape}')
    print(f'Shape of the output data: {targets.shape}')

    classifier = MLPClassifier(random_state=0, verbose=1)
    classifier.fit(inputs, targets)

    # Testing the model
    test = []
    actual = []
    for emotion in tqdm(emotions, desc='Extracting test data'):
        path = os.path.join(os.path.dirname(__file__), "faces", "test", emotion)
        for file in os.listdir(path):
            with Image.open(os.path.join(path, file)) as image:
                test.append(np.array(image).flatten())
            actual.append(emotions.index(emotion))

    test = np.array(test)

    print(f'Shape of the test data:  {test.shape}')

    results = classifier.predict(test)

    correct = 0
    for i in range(len(results)):
        if results[i] == actual[i]:
            correct += 1

    print(f'The model was tested: {round(correct/len(results), 3) * 100}% accuracy')

if __name__ == '__main__':
    learn_emotions()