import os
import time
import pickle
import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from torch.utils.data import TensorDataset, DataLoader

DATA_DIR = Path(__file__).parent / 'faces'
TRAIN_DIR = DATA_DIR / 'train'
TEST_DIR = DATA_DIR / 'test'
if not TRAIN_DIR.exists():
    raise ValueError(f'Path {str(TRAIN_DIR)} does not exist')
if not TEST_DIR.exists():
    raise ValueError(f'Path {str(TEST_DIR)} does not exist')
if not (set(p.name for p in TEST_DIR.iterdir() if p.is_dir()) <= set(p.name for p in TRAIN_DIR.iterdir() if p.is_dir())):
    raise ValueError('There are `test` labels that do not exist in `train`')
EMOTIONS = sorted((p.name for p in TRAIN_DIR.iterdir() if p.is_dir()))
MODEL_ID = 'openai/clip-vit-base-patch16'

parser = argparse.ArgumentParser(
    description="Train, test and save a Random Forest model for the use of detecting a certain emotion in a human face"
)
parser.add_argument('--estimators', '-e', type=int, default=1000, help='The amount of estimators in the Random Forest model')
parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size to feed encoder to produce vector embeddings')
ARGS = parser.parse_args()

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

def get_data(
        directory,
        model=AutoModel.from_pretrained(MODEL_ID),
        processor=AutoProcessor.from_pretrained(MODEL_ID)
    ):

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    preproc_filename = DATA_DIR / "preprocessed_data" / f'preprocessed_{directory.name}_data.npz'
    if preproc_filename.exists():
        print(f'Loading data from "{preproc_filename}"... ', end='')
        npz = np.load(preproc_filename)
        img_vecs = npz['img_vecs']
        targets = npz['targets']
        print('Done!')
    else:
        imgs = np.stack([
            np.array(Image.open(filename).convert("RGB")).transpose(2, 0, 1)
            for emotion in tqdm(EMOTIONS, desc=f'Extracting {directory.name} data')
            if (directory / emotion).exists()
            for filename in sorted((directory / emotion).iterdir())
            if filename.suffix == '.jpg'
        ])
        targets = np.array([
            label
            for label, emotion in enumerate(EMOTIONS)
            if (directory / emotion).exists()
            for filename in sorted((directory / emotion).iterdir())
            if filename.suffix == '.jpg'
        ])
        dataset = DataLoader(
            TensorDataset(torch.from_numpy(imgs)),
            batch_size=ARGS.batch_size,
        )
        with torch.inference_mode():
            # breakpoint()
            # img_vecs = torch.cat([
            #     model.get_image_features(**processor(images=img_batch.to(device), return_tensors='pt'))
            #     for (img_batch,) in tqdm(dataset, desc='Producing image vectors')
            # ])
            img_vecs = torch.cat([
                model.get_image_features(**processor(images=img_batch, return_tensors='pt'))
                for (img_batch,) in tqdm(dataset, desc='Producing image vectors')
            ])
        img_vecs = np.array(img_vecs)
        os.mkdir("faces" / "preprocessed_data")
        np.savez_compressed(preproc_filename, img_vecs=img_vecs, targets=targets)
    return img_vecs, targets

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
        n_estimators=ARGS.estimators
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
    if not os.path.isdir("models"):
        os.mkdir("models")
    with open(f"./models/random_forest_{ARGS.estimators}.pickle", 'wb') as file:
        pickle.dump(classifier, file)

if __name__ == '__main__':

    # Start timer
    start = time.perf_counter()

    # Train the model on training data
    train_inputs, train_targets = get_data(TRAIN_DIR)
    test_inputs, test_targets = get_data(TEST_DIR)
    emotion_ai = train(train_inputs, train_targets)

    # Test the model on training data
    evaluate(emotion_ai, train_inputs, train_targets, 'Train')

    # Test the model on testing data
    evaluate(emotion_ai, test_inputs, test_targets, 'Test')

    # Display the results of tests
    plt.show()

    # Save the model
    save_classifier(emotion_ai)

    # Stop timer and calculate elapsed time
    print(f'Emotion model trained, tested and saved in {format_time(round(time.perf_counter() - start))}')
