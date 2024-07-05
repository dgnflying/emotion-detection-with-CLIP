import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

from create_embeddings import (
    PREPROC_IMGS_DIR,
    RAW_TRAIN_DIR,
    RAW_TEST_DIR,
    create_image_embeddings,
)

parser = argparse.ArgumentParser(
    description="Train, test and save an AI model for the use of detecting a certain emotion in a human face"
)
parser.add_argument(
    "--hidden_layers",
    "-l",
    type=int,
    default=[100],
    help="The hidden layers of the model",
    nargs="+",
)
parser.add_argument(
    "--learning_rate",
    "-r",
    type=float,
    default=0.001,
    help="The learning rate of the model",
)
parser.add_argument(
    "--batch_size",
    "-b",
    type=int,
    default=64,
    help="The batch size for training the model",
)
parser.add_argument(
    "--max_iter",
    "-m",
    type=int,
    default=200,
    help="The maximum number of iterations for training the model",
)
ARGS = parser.parse_args()
HIDDEN_LAYERS = tuple(ARGS.hidden_layers)


def get_data(directory):
    preproc_filename = PREPROC_IMGS_DIR / f"{directory.name}.npz"
    if preproc_filename.exists():
        print(f'Loading data from "{preproc_filename}"... ', end="")
        npz = np.load(preproc_filename)
        vecs = npz["vecs"]
        targets = npz["targets"]
        print("Done!")
        return vecs, targets
    else:
        return create_image_embeddings(directory, 32)


def train(inputs, targets):
    # Training the model
    print(
        f'Training a "Batch Size = {ARGS.batch_size}", "Hidden Layers = {HIDDEN_LAYERS}" and "Learning Rate = {ARGS.learning_rate}" model on {len(train_inputs)} image vectors...'
    )
    return MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        batch_size=ARGS.batch_size,
        learning_rate_init=ARGS.learning_rate,
        max_iter=ARGS.max_iter,
        random_state=0,
        verbose=True,
    ).fit(inputs, targets)


def evaluate(classifier, inputs, targets, partition):
    # Testing the model
    results = classifier.predict(inputs)
    cm = confusion_matrix(targets, results)
    accuracy = f"{(results == targets).mean() * 100:.3f}%"
    print(f"Accuracy on {partition.lower()}ing data: {accuracy}")
    return cm, accuracy


def save_data(classifier, train_cm, test_cm, train_results, test_results):

    # Create a model folder
    OUTPUT_DIR = Path("output")
    if not OUTPUT_DIR.is_dir():
        OUTPUT_DIR.mkdir()
    model_iter = 0
    date_str = time.strftime("%Y-%m-%d")
    while (MODEL_DIR := OUTPUT_DIR / f"{date_str}-{model_iter}").is_dir():
        model_iter += 1
    MODEL_DIR.mkdir()

    # Save the model
    model_filename = MODEL_DIR / "emotion_ai_model.pickle"
    with open(model_filename, "wb") as model_file:
        pickle.dump(classifier, model_file)
    print(f'Model saved to "{model_filename}"')

    # Save the model's hyperparameters and accuracy results
    results_filename = MODEL_DIR / "data.json"
    results = {
        "hyperparameters": classifier.get_params(),
        "training-results": train_results,
        "testing-results": test_results,
    }
    with open(results_filename, "w") as results_file:
        json.dump(results, results_file)
    print(f'Results saved to "{results_filename}"')

    # Save the confusion matrix data
    cm_folder = MODEL_DIR / "confusion_matrices.npz"
    np.savez_compressed(cm_folder, train_cm=train_cm, test_cm=test_cm)
    print(f'Confusion matrices saved to "{cm_folder}"')

    # Save the loss curve
    losses_filename = MODEL_DIR / "loss_curve.npz"
    np.savez_compressed(losses_filename, loss_curve=np.array(classifier.loss_curve_))
    print(f'Loss curve saved to "{losses_filename}"')

    return MODEL_DIR


if __name__ == "__main__":

    # Train the model on training data
    train_inputs, train_targets = get_data(RAW_TRAIN_DIR)
    emotion_ai = train(train_inputs, train_targets)

    # Test the model on training data
    train_cm, train_accuracy = evaluate(
        emotion_ai, train_inputs, train_targets, "Train"
    )

    # Test the model on testing data
    test_inputs, test_targets = get_data(RAW_TEST_DIR)
    test_cm, test_accuracy = evaluate(emotion_ai, test_inputs, test_targets, "Test")

    # Save the model
    MODEL_DIR = save_data(
        emotion_ai,
        train_cm=train_cm,
        test_cm=test_cm,
        train_results=train_accuracy,
        test_results=test_accuracy,
    )

    # Display the result graphs
    os.system(f"py plot_data.py -f {MODEL_DIR.name} -l -c")
