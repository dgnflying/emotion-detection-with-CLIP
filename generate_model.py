import time
import pickle
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from create_embeddings import EMOTIONS, PREPROC_IMGS_DIR, RAW_TRAIN_DIR, RAW_TEST_DIR, create_image_embeddings

parser = argparse.ArgumentParser(
    description="Train, test and save an AI model for the use of detecting a certain emotion in a human face"
)
parser.add_argument('--no_save', '-s', help='Opt out of saving the model', action=argparse.BooleanOptionalAction)
parser.add_argument('--hidden_layers', '-l', type=int, default=[100], help='The hidden layers of the model', nargs='+')
parser.add_argument('--batch_size', '-b', type=int, default=200, help='The batch size for training the model')
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
    preproc_filename = PREPROC_IMGS_DIR / f'{directory.name}.npz'
    if preproc_filename.exists():
        print(f'Loading data from "{preproc_filename}"... ', end='')
        npz = np.load(preproc_filename)
        vecs = npz['vecs']
        targets = npz['targets']
        print('Done!')
        return vecs, targets
    else:
        return create_image_embeddings(directory, 32)

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
        batch_size=ARGS.batch_size,
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
    MODELS_DIR = Path("output/models")
    if not MODELS_DIR.is_dir():
        MODELS_DIR.mkdir(parents=True)
    model_iter = 0
    while (model_filename := MODELS_DIR / f"emotion_ai_{model_iter}.pickle").exists():
        model_iter += 1
    with open(model_filename, 'wb') as model_file:
        pickle.dump(classifier, model_file)
    print(f'Model saved to "{model_filename}"')

    # Save the loss curve
    DATA_DIR = Path("output/data")
    if not DATA_DIR.is_dir():
        DATA_DIR.mkdir(parents=True)
    losses_filename = DATA_DIR / f"emotion_ai_{model_iter}_loss_curve.npz"
    np.savez_compressed(losses_filename, np.array(classifier.loss_curve_))
    print(f'Loss curve saved to "{losses_filename}"')

if __name__ == '__main__':

    # Start timer
    start_time = time.perf_counter()

    # Train the model on training data
    train_inputs, train_targets = get_data(RAW_TRAIN_DIR)
    emotion_ai = train(train_inputs, train_targets)

    # Test the model on training data
    evaluate(emotion_ai, train_inputs, train_targets, 'Train')

    # Test the model on testing data
    test_inputs, test_targets = get_data(RAW_TEST_DIR)
    evaluate(emotion_ai, test_inputs, test_targets, 'Test')

    # Save the model
    if not ARGS.no_save:
        save_classifier(emotion_ai)

    stop_time = time.perf_counter()

    # Display the results of tests
    plt.show()

    # Stop timer and calculate elapsed time
    print(
        f"Emotion model trained{', tested and saved' if not ARGS.no_save else ' and tested'} in {format_time(round(stop_time - start_time))}"
    )