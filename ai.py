from sklearn.neural_network import MLPClassifier
from PIL import Image
import numpy as np
import os
import random

emotions = ["angry", "disgusted", "happy", "neutral", "sad", "scared", "surprised"]

def learn_emotions():
    # Training the model
    inputs = []
    targets = []
    for emotion in emotions:
        path = os.path.join(os.path.dirname(__file__), "faces", "train", emotion)
        for file in os.listdir(path):
            with Image.open(os.path.join(path, file)) as image:
                inputs.append(np.array(image).flatten())
            targets.append(emotions.index(emotion))
    inputs = np.array(inputs)
    targets = np.array(targets)

    print(f'Shape of the input data:  {inputs.shape}')
    print(f'Shape of the output data: {targets.shape}')

    classifier = MLPClassifier(random_state=0)
    classifier.fit(inputs, targets)

    # Testing the model
    test = []
    test_file = []
    actual = []
    for emotion in emotions:
        path = os.path.join(os.path.dirname(__file__), "faces", "test", emotion)
        images = os.listdir(path)
        random_image = images[random.randrange(0, len())]
        with Image.open(os.path.join(path, random_image)) as image:
            test.append(np.array(image).flatten())
        test_file.append(random_image)
        actual.append(emotions.index(emotion))

    test = np.array(test)

    print(f'Shape of the test data:  {test.shape}')

    results = classifier.predict(test)

    for i in range(len(results)):
        print(f'NN predicts {emotions[results[i]].upper()} (It was {emotions[actual[i]].upper()})')
        print('Image: ' + os.path.join(os.path.dirname(__file__), "faces", "test", emotions[actual[i]], test_file[i]))
        print()

if __name__ == '__main__':
    learn_emotions()