import time
import pickle
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from preprocess_images import EMOTIONS, DATA_DIR, TRAIN_DIR, TEST_DIR, preprocess_images

parser = argparse.ArgumentParser(
    description="Train, test and save an AI model for the use of detecting a certain emotion in a human face"
)
parser.add_argument('--save_model', '-s', help='Save the model', action=argparse.BooleanOptionalAction)
parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size to feed encoder to produce vector embeddings')
parser.add_argument('--hidden_layers', '-l', type=int, default=[100], help='Hidden layers for the model', nargs='+')
ARGS = parser.parse_args()
HIDDEN_LAYERS = tuple(ARGS.hidden_layers)

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

def get_data(directory):
    preproc_filename = DATA_DIR / "preprocessed_data" / f'preprocessed_{directory.name}_data.npz'
    if preproc_filename.exists():
        print(f'Loading data from "{preproc_filename}"... ', end='')
        npz = np.load(preproc_filename)
        img_vecs = npz['img_vecs']
        targets = npz['targets']
        print('Done!')
        return img_vecs, targets
    else:
        return preprocess_images(directory, ARGS.batch_size)

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
    print(f'Training a Hidden Layers: {HIDDEN_LAYERS} model on {len(train_inputs)} image vectors...')
    return MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        alpha=0.0001,
        batch_size='auto',
        learning_rate_init=0.001,
        max_iter=200,
        random_state=0,
        verbose=True,
    ).fit(inputs, targets)

def evaluate(classifier, inputs, targets, partition):
    # Testing the model
    results = classifier.predict(inputs)
    display_data(predictions=results, targets=targets, labels=EMOTIONS, title=f"{partition.upper()} Data")
    print(
        f'Accuracy on {partition.lower()}ing data: '
        f'{(results == targets).mean() * 100:.3f}%'
    )

def save_classifier(classifier):
    # Saving the model
    MODELS_DIR = Path("models")
    if not MODELS_DIR.is_dir():
        MODELS_DIR.mkdir()
    model_iter = 0
    while (model_filename := MODELS_DIR / f"emotion_ai_{model_iter}.pickle").exists():
        model_iter += 1
    with open(model_filename, 'wb') as model_file:
        pickle.dump(classifier, model_file)

if __name__ == '__main__':

    # Start timer
    start_time = time.perf_counter()

    # Train the model on training data
    train_inputs, train_targets = get_data(TRAIN_DIR)
    emotion_ai = train(train_inputs, train_targets)

    # Test the model on training data
    evaluate(emotion_ai, train_inputs, train_targets, 'Train')

    # Test the model on testing data
    test_inputs, test_targets = get_data(TEST_DIR)
    evaluate(emotion_ai, test_inputs, test_targets, 'Test')

    # Save the model
    if ARGS.save_model:
        save_classifier(emotion_ai)

    stop_time = time.perf_counter()

    # Display the results of tests
    plt.show()

    # Stop timer and calculate elapsed time
    print(
        f"Emotion model trained{', tested and saved' if ARGS.save_model == True else ' and tested'} in {format_time(round(stop_time - start_time))}"
    )