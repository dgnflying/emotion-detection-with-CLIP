from sklearn.neural_network import MLPClassifier
from PIL import Image
import numpy as np
import os
import random

emotions = ["angry", "disgusted", "happy", "neutral", "sad", "scared", "surprised"]

def learn_digits():
    # Training the model
    inputs = []
    targets = []
    for emotion in emotions:
        path = os.path.join(os.path.dirname(__file__), "faces", "train", emotion)
        for filename in os.listdir(path):
            with Image.open(os.path.join(path, filename)) as image:
                inputs.append(np.array(image).flatten())
            targets.append(emotions.index(emotion))
    inputs = np.array(inputs)
    targets = np.array(targets)

    print(f'Shape of the input data:  {inputs.shape}')
    print(f'Shape of the output data: {targets.shape}')

    classifier = MLPClassifier(random_state=0)
    classifier.fit(inputs, targets)

    # Testing the model
    test_large = []
    actual = []
    test_filenames = []
    for emotion in emotions:
        path = os.path.join(os.path.dirname(__file__), "faces", "test", emotion)
        for filename in os.listdir(path):
            with Image.open(os.path.join(path, filename)) as image:
                test_large.append(np.array(image).flatten())
                test_filenames.append(filename)
            actual.append(emotions.index(emotion))

    random_indexes = random.sample(range(len(test_large)), 10)
    test = [test_large[i] for i in random_indexes]
    actual = [actual[i] for i in random_indexes]
    test_filenames = [test_filenames[i] for i in random_indexes]

    test = np.array(test)
    actual = np.array(actual)

    print(f'Shape of the test data:  {test.shape}')
    print(f'Shape of the actual data: {actual.shape}')

    results = classifier.predict(test)

    for i in range(len(results)):
        print(f'NN predicts {emotions[results[i]].upper()} (It was {emotions[actual[i]].upper()})')
        print('Image: ' + os.path.join(os.path.dirname(__file__), "faces", "test", emotions[actual[i]], test_filenames[i]))
        print()

if __name__ == '__main__':
    learn_digits()