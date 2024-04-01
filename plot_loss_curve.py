import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Plot the loss curve of a classifier')
parser.add_argument('--file', '-f', type=str, help='The file containing the loss curve data')
ARGS = parser.parse_args()

def plot_loss_curve(loss_curve_file):

    # Load the loss curve data
    data = np.load(Path("output") / loss_curve_file / "model_loss_curve.npz")
    loss_curve = data['loss_curve']

    # Plot the loss curve
    plt.plot(loss_curve)
    plt.title('Loss Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

if __name__ == '__main__':
    plot_loss_curve(ARGS.file)