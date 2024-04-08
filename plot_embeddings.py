import argparse
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from create_embeddings import PREPROC_TEXT_DIR, PREPROC_IMGS_DIR, RAW_TRAIN_DIR, EMOTIONS, create_text_embeddings, create_image_embeddings

parser = argparse.ArgumentParser(
    description="Plot the embeddings of the images in the dataset to visualize the distribution of the data."
)
parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size to feed encoder to produce vector embeddings')
parser.add_argument('--average', '-a', help='Display the average vector for each emotion', action=argparse.BooleanOptionalAction)
parser.add_argument('--all', '-A', help='Display all embeddings for every emotion', action=argparse.BooleanOptionalAction)
parser.add_argument('--text', '-t', help='Display the text embeddings of the emotions', action=argparse.BooleanOptionalAction)
parser.add_argument(
    '--comparison',
    '-c',
    help="Display comparisons between each emotion's average image vector and their text counterpart",
    action=argparse.BooleanOptionalAction
)
parser.add_argument('--titles', '-T', help='Display the titles of the plots', action=argparse.BooleanOptionalAction)
ARGS = parser.parse_args()

FIG_SIZE = (10, 8)

COLORS = {
    'angry': {'text': sns.light_palette("red")[3], 'image': sns.dark_palette("red")[3]},
    'disgust': {'text': sns.light_palette("green")[3], 'image': sns.dark_palette("green")[3]},
    'fear': {'text': sns.light_palette("purple")[3], 'image': sns.dark_palette("purple")[3]},
    'happy': {'text': sns.light_palette("yellow")[3], 'image': sns.dark_palette("yellow")[3]},
    'neutral': {'text': sns.light_palette("gray")[3], 'image': sns.dark_palette("gray")[3]},
    'sad': {'text': sns.light_palette("blue")[3], 'image': sns.dark_palette("blue")[3]},
    'surprise': {'text': sns.light_palette("orange")[3], 'image': sns.dark_palette("orange")[3]},
}

def get_embeddings(type, directory):
    if type == 'IMGS':
        preproc_filename = PREPROC_IMGS_DIR / f'{directory.name}.npz'
    elif type == 'TEXT':
        preproc_filename = PREPROC_TEXT_DIR / f'{directory.name}.npz'
    if preproc_filename.exists():
        print(f'Loading data from "{preproc_filename}"... ', end='')
        npz = np.load(preproc_filename)
        vecs = npz['vecs']
        targets = npz['targets']
        print('Done!')
        return vecs, targets
    else:
        if type == 'IMGS':
            return create_image_embeddings(directory, ARGS.batch_size)
        elif type == 'TEXT':
            return create_text_embeddings(directory)

if __name__ == '__main__':
    # Reduce embeddings to 2 dimensions
    img_vecs, img_targets = get_embeddings('IMGS', RAW_TRAIN_DIR)
    img_data = TSNE(random_state=0, verbose=1).fit_transform(img_vecs)

    # Plot all embeddings for every emotion
    if ARGS.all:
        plt.figure(figsize=FIG_SIZE)
        if ARGS.titles:
            plt.suptitle('Scatter Plot of Emotions')
        for i, emotion in enumerate(EMOTIONS):
            color_image = COLORS[emotion]['image']
            indices = img_targets == i
            plt.scatter(img_data[indices, 0], img_data[indices, 1], label=emotion, color=color_image)
            if ARGS.titles:
                plt.title('t-SNE Visualization of the "Train" Image Embeddings')
            plt.legend()

    # Plot average vector for each emotion
    if ARGS.average:
        plt.figure(figsize=FIG_SIZE)
        if ARGS.titles:
            plt.suptitle('Average Vector of Emotions')
        for i, emotion in enumerate(EMOTIONS):
            color_image = COLORS[emotion]['image']
            indices = img_targets == i
            avg_vec = np.mean(img_data[indices], axis=0)
            plt.scatter(avg_vec[0], avg_vec[1], label=emotion, color=color_image)
            plt.plot([0, avg_vec[0]], [0, avg_vec[1]], color=color_image)
        if ARGS.titles:
            plt.title('t-SNE Visualization of the average "Train" Image Embeddings')
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        plt.legend()

    # Plot the text vectors
    if ARGS.text:
        text_vecs, text_targets = get_embeddings('TEXT', PREPROC_TEXT_DIR)
        text_data = TSNE(random_state=0, verbose=1, perplexity=5).fit_transform(text_vecs)

        plt.figure(figsize=FIG_SIZE)
        if ARGS.titles:
            plt.suptitle('Scatter Plot of Text Emotions')
        for i, emotion in enumerate(EMOTIONS):
            color_text = COLORS[emotion]['text']
            indices = text_targets == i
            plt.scatter(text_data[indices, 0], text_data[indices, 1], label=emotion, color=color_text)
            plt.plot([0, text_data[indices, 0][0]], [0, text_data[indices, 1][0]], color=color_text)
            if ARGS.titles:
                plt.title('t-SNE Visualization of Text Embeddings')
            plt.axhline(0, color='black')
            plt.axvline(0, color='black')
            plt.legend()


    # Plot comparisons between each emotion's average image vector and their text counterpart
    if ARGS.comparison:

        text_vecs, text_targets = get_embeddings('TEXT', PREPROC_TEXT_DIR)
        text_data = TSNE(random_state=0, verbose=1, perplexity=5).fit_transform(text_vecs)

        plt.figure(figsize=FIG_SIZE)
        for i, emotion in enumerate(EMOTIONS):
            color_text = COLORS[emotion]['text']
            color_image = COLORS[emotion]['image']
            text_indices = text_targets == i
            avg_img_vec = np.mean(img_data[img_targets == i], axis=0)
            plt.scatter(text_data[text_indices, 0], text_data[text_indices, 1], label=f'{emotion.capitalize()} (Text)', color=color_text)
            plt.plot([0, text_data[text_indices, 0][0]], [0, text_data[text_indices, 1][0]], color=color_text)
            plt.scatter(avg_img_vec[0], avg_img_vec[1], label=f'{emotion.capitalize()} (Image)', color=color_image)
            plt.plot([0, avg_img_vec[0]], [0, avg_img_vec[1]], color=color_image)
        if ARGS.titles:
            plt.title(
                't-SNE Visualization of the average of the "Train" Image Embeddings compared with their Text Counterparts'
            )
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        plt.legend()

    plt.show()