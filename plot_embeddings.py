import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from preprocess_images import DATA_DIR, TRAIN_DIR, TEST_DIR, EMOTIONS, preprocess_images

parser = argparse.ArgumentParser(
    description="Plot the embeddings of the images in the dataset to visualize the distribution of the data."
)
parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size to feed encoder to produce vector embeddings')
ARGS = parser.parse_args()

FIG_SIZE = (10, 8)

def get_embeddings(directory):
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

if __name__ == '__main__':
    # Reduce embeddings to 2 dimensions
    train_img_vecs, train_targets = get_embeddings(TRAIN_DIR)
    test_img_vecs, test_targets = get_embeddings(TEST_DIR)
    train_2d = TSNE(random_state=0, verbose=1).fit_transform(train_img_vecs)
    test_2d = TSNE(random_state=0, verbose=1).fit_transform(test_img_vecs)

    # Plot the embeddings
    plt.figure(figsize=FIG_SIZE)
    for i, emotion in enumerate(EMOTIONS):
        indices = train_targets == i
        plt.scatter(train_2d[indices, 0], train_2d[indices, 1], label=emotion)
    plt.title('t-SNE Visualization of "Train" Image Embeddings')
    plt.legend()

    plt.figure(figsize=FIG_SIZE)
    for i, emotion in enumerate(EMOTIONS):
        indices = test_targets == i
        plt.scatter(test_2d[indices, 0], test_2d[indices, 1], label=emotion)
    plt.title('t-SNE Visualization of "Test" Image Embeddings')
    plt.legend()

    plt.show()