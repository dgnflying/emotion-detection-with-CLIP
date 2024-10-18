import os
import argparse
import json
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.metrics import confusion_matrix, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Train and test FER2013 AI models using different architectures"
)

parser.add_argument(
    "--architecture",
    "-a",
    choices=["mlp", "rf", "cnn"],
    default="mlp",
    help="The architecture of the model",
)

# RF arguments
parser.add_argument(
    "--n_estimators",
    "-n",
    type=int,
    default=100,
    help="Number of trees in the random forest",
)

# MLP / CNN arguments

parser.add_argument(
    "--batch_size",
    "-bs",
    type=int,
    default=64,
    help="Batch size for training the model",
)

parser.add_argument(
    "--learning_rate",
    "-lr",
    type=float,
    default=0.001,
    help="Learning rate for training the model",
)

# MLP arguments
parser.add_argument(
    "--hidden_units_1",
    "-hu1",
    type=int,
    default=128,
    help="Number of hidden units in the first layer of the MLP",
)

parser.add_argument(
    "--hidden_units_2",
    "-hu2",
    type=int,
    default=64,
    help="Number of hidden units in the second layer of the MLP",
)

# CNN arguments
parser.add_argument(
    "--conv1_filters",
    "-c1",
    type=int,
    default=32,
    help="Number of filters in the first convolutional layer of the CNN",
)

parser.add_argument(
    "--conv2_filters",
    "-c2",
    type=int,
    default=64,
    help="Number of filters in the second convolutional layer of the CNN",
)

parser.add_argument(
    "--fc1_units",
    "-fc1",
    type=int,
    default=128,
    help="Number of units in the fully connected layer of the CNN",
)

parser.add_argument(
    "--kernel_size",
    "-k",
    type=int,
    default=3,
    help="Kernel size for the convolutional layers of the CNN",
)

ARGS = parser.parse_args()
DATA_PATH = os.path.join(os.getcwd(), "data")
EPOCHS = 10
LABEL_MAP = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
IMAGE_WIDTH = 48


class MLP(nn.Module):
    def __init__(
        self,
        hidden_units_1,
        hidden_units_2,
    ):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(IMAGE_WIDTH**2, hidden_units_1)
        self.fc2 = nn.Linear(hidden_units_1, hidden_units_2)
        self.fc3 = nn.Linear(hidden_units_2, 7)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(
        self,
        conv1_filters,
        conv2_filters,
        fc1_units,
        kernel_size,
        padding=1,
    ):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            1,
            conv1_filters,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2 = nn.Conv2d(
            conv1_filters,
            conv2_filters,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.fc1 = nn.Linear(
            self._output_size(kernel_size, padding) * conv2_filters,
            fc1_units,
        )
        self.fc2 = nn.Linear(fc1_units, 7)

    def _output_size(self, kernel_size, padding):
        # Assuming square input
        # Conv1
        size = (IMAGE_WIDTH - kernel_size + 2 * padding // 1) + 1
        # Pool1
        size = size // 2
        # Conv2
        size = (size - kernel_size + 2 * padding // 1) + 1
        # Pool2
        size = size // 2
        # Final dimensions
        return size * size

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_raw_data():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Adjust for grayscale images
        ]
    )
    train_set = datasets.FER2013(root=DATA_PATH, transform=transform, split="train")
    test_set = datasets.FER2013(root=DATA_PATH, transform=transform, split="test")

    return train_set, test_set


def load_embedding_data():
    train_embeddings_file = os.path.join(
        DATA_PATH, "fer2013", "embedding_data", "train_image_embeddings.npz"
    )
    test_embeddings_file = os.path.join(
        DATA_PATH, "fer2013", "embedding_data", "test_image_embeddings.npz"
    )
    if os.path.exists(train_embeddings_file) and os.path.exists(test_embeddings_file):

        print(f'Loading train data from "{train_embeddings_file}"... ', end="")
        train_embeddings = np.load(train_embeddings_file)
        train_vecs = train_embeddings["vecs"]
        train_targets = train_embeddings["targets"]
        print("Done!")

        print(f'Loading test data from "{test_embeddings_file}"... ', end="")
        test_embeddings = np.load(test_embeddings_file)
        test_vecs = test_embeddings["vecs"]
        test_targets = test_embeddings["targets"]
        print("Done!")

        return train_vecs, train_targets, test_vecs, test_targets
    else:
        exit()


def run_model_raw(model):

    # Load data
    train_set, test_set = load_raw_data()

    if ARGS.architecture == "mlp" or ARGS.architecture == "cnn":

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=ARGS.learning_rate, momentum=0.9)

        print(f"Training the model for {EPOCHS} epochs...")
        loss_curve = []
        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            epoch_steps = 0

            # Training the model
            for data in tqdm(train_set, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
                # sends the inputs to the gpu
                image, label = data
                image, label = image.to(device), torch.tensor(label).unsqueeze(0).to(
                    device
                )

                # zero the parameter gradients
                for param in model.parameters():
                    param.grad = None

                # forward + backward + optimize
                outputs = model(image)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

                # print statistics
                epoch_loss += loss.item()
                epoch_steps += 1

                loss = epoch_loss / epoch_steps
                loss_curve.append(loss)

                # if epoch_steps % 500 == 499:
                #     print(
                #         f"Epoch {epoch + 1}, Sample {epoch_steps:5d}/{len(train_set) - 1} - loss: {loss:.3f}"
                #     )

                # if epoch_steps == len(train_set) - 1:
                #     print(
                #         f"Epoch {epoch + 1}, Sample {len(train_set) - 1}/{len(train_set) - 1} - loss: {epoch_loss / epoch_steps:.3f}"
                #     )

            print(f"Finished Epoch {epoch + 1}/{EPOCHS}")

        # Evaluating the model
        print("Evaluating Train Data...", end="")
        loss = 0.0
        epoch_steps = 0
        total = 0
        correct = 0
        train_results = []
        train_labels = []

        model.eval()

        for data in tqdm(train_set, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            with torch.no_grad():
                image, label = data
                image, label = image.to(device), torch.tensor(label).unsqueeze(0).to(
                    device
                )

                train_labels.append(label.cpu())

                outputs = model(image)
                _, predicted = torch.max(outputs.data, 1)
                train_results.append(predicted.cpu().numpy())
                total += label.size(0)
                correct += (predicted == label).sum().item()

                loss += criterion(outputs, label).cpu().numpy()
                epoch_steps += 1
        print("Done!")

        train_loss = loss / epoch_steps
        train_accuracy = correct / total

        print(f"Train Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.3f}")

        train_cm = confusion_matrix(np.array(train_labels), np.array(train_results))

        print("Evaluating Test Data...", end="")
        test_loss = 0.0
        test_epoch_steps = 0
        test_total = 0
        test_correct = 0
        test_results = []
        test_labels = []
        for image in tqdm(train_set, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            with torch.no_grad():
                image, label = data
                image, label = image.to(device), torch.tensor(label).unsqueeze(0).to(
                    device
                )

                test_labels.append(label)

                outputs = model(image)
                _, predicted = torch.max(outputs.data, 1)
                test_results.append(predicted.cpu().numpy())
                test_total += label.size(0)
                test_correct += (predicted == label).sum().item()

                test_loss += criterion(outputs, label).cpu().numpy()
                test_epoch_steps += 1
        print("Done!")

        test_loss = test_loss / test_epoch_steps
        test_accuracy = test_correct / test_total

        print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}")

        test_labels = np.array(test_labels)
        test_results = torch.cat(torch.tensor(test_results))
        test_cm = confusion_matrix(test_labels, test_results)

        loss_curve = np.array(loss_curve)

        return train_accuracy, test_accuracy, train_cm, test_cm, loss_curve

    elif ARGS.architecture == "rf":
        # Initialize lists to store samples and labels
        samples = []
        labels = []
        for sample, label in tqdm(train_set, desc="Sorting train data"):
            samples.append(sample)
            labels.append(label)
        samples = torch.stack(samples).flatten(1)
        labels = torch.tensor(labels)

        # Train the model
        print("Building forest...")
        model.fit(samples, labels)
        print("Done!")

        # Test the model
        test_samples = []
        test_labels = []
        for sample, label in tqdm(test_set, desc="Sorting test data"):
            test_samples.append(sample)
            test_labels.append(label)
        test_samples = torch.stack(test_samples).flatten(1)
        test_labels = torch.tensor(test_labels)

        print("Evaluating the model...", end="")
        outputs = model.predict(samples)
        accuracy = f"{(outputs == labels).mean() * 100:.3f}%"
        test_outputs = model.predict(test_samples)
        test_accuracy = f"{(test_outputs == test_labels).mean() * 100:.3f}%"
        print("Done!")

        print(f"Train Accuracy: {accuracy:.3f}")
        print(f"Test Accuracy: {test_accuracy:.3f}")

        train_cm = confusion_matrix(labels, outputs)
        test_cm = confusion_matrix(test_labels, test_outputs)

        return accuracy, test_accuracy, train_cm, test_cm, model.loss_curve_

    print(f"Finished {ARGS.architecture} epochs")


def run_model_embeddings(model):

    # Load data
    print("Loading data...", end="")
    train_vecs, train_targets, test_vecs, test_targets = load_embedding_data()
    print("Done!")

    if ARGS.architecture == "mlp" or ARGS.architecture == "cnn":

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=ARGS.learning_rate, momentum=0.9)

        # Initialize data loaders
        print("Initializing data loaders...", end="")
        train_loader = torch.utils.data.DataLoader(
            train_vecs, batch_size=ARGS.batch_size, shuffle=True, num_workers=8
        )
        test_loader = torch.utils.data.DataLoader(
            test_vecs, batch_size=ARGS.batch_size, shuffle=True, num_workers=8
        )
        print("Done!")

        loss_curve = []
        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            epoch_steps = 0

            # Training the model
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                for param in model.parameters():
                    param.grad = None

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # statistics
                epoch_loss += loss.item()
                epoch_steps += 1
                loss = epoch_loss / epoch_steps
                loss_curve.append(loss)

                if i % 500 == 499:
                    print(
                        f"Epoch {epoch + 1}, Sample {i + 1:5d} - loss: {epoch_loss / epoch_steps:.3f}"
                    )

                if i == len(train_loader) - 1:
                    print(
                        f"Epoch {epoch + 1}, Final loss - {epoch_loss / epoch_steps:.3f}"
                    )

            print(f"Finished Epoch {epoch + 1} ...Restarting")

        # Evaluating the model
        print("Evaluating Train Data...", end="")
        loss = 0.0
        epoch_steps = 0
        total = 0
        correct = 0
        train_results = []
        train_labels = []

        model.eval()

        for i, data in enumerate(train_loader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                train_labels.append(labels.cpu().numpy())

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                train_results.append(predicted.cpu().numpy())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss += criterion(outputs, labels).cpu().numpy()
                epoch_steps += 1
        print("Done!")

        print("Evaluating Test Data...", end="")
        test_loss = 0.0
        test_epoch_steps = 0
        test_total = 0
        test_correct = 0
        test_results = []
        test_labels = []
        for i, data in enumerate(test_loader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                test_labels.append(labels.cpu().numpy())

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_results.append(predicted.cpu().numpy())
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                test_loss += criterion(outputs, labels).cpu().numpy()
                test_epoch_steps += 1
        print("Done!")

        train_loss = loss / epoch_steps
        train_accuracy = correct / total

        test_loss = test_loss / test_epoch_steps
        test_accuracy = test_correct / test_total

        print(f"Train Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.3f}")
        print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}")

        train_cm = confusion_matrix(torch.cat(train_labels), torch.cat(train_results))
        test_cm = confusion_matrix(torch.cat(test_labels), torch.cat(test_results))

        loss_curve = np.array(loss_curve)

        return train_accuracy, test_accuracy, train_cm, test_cm, loss_curve

    elif ARGS.architecture == "rf":
        # Train the model
        print("Building forest...")
        model.fit(train_vecs, train_targets)
        print("Done!")

        # Evaluate the model
        print("Evaluating the model...", end="")
        outputs = model.predict(train_vecs)
        accuracy = f"{(outputs == train_targets).mean() * 100:.3f}%"
        test_outputs = model.predict(test_vecs)
        test_accuracy = f"{(test_outputs == test_targets).mean() * 100:.3f}%"
        print("Done!")

        print(f"Train Accuracy: {accuracy}")
        print(f"Test Accuracy: {test_accuracy}")

        train_cm = confusion_matrix(train_targets, outputs)
        test_cm = confusion_matrix(test_targets, test_outputs)

        return accuracy, test_accuracy, train_cm, test_cm, model.loss_curve_

    print(f"Finished {ARGS.architecture} epochs")


if __name__ == "__main__":

    if ARGS.architecture == "mlp":
        print(
            f"Comparing CLIP effectivity using two multi-layer perceptron models with a hidden unit structure of ( inputs => {ARGS.hidden_units_1} => {ARGS.hidden_units_2} => output )"
        )
        model_raw = MLP(
            hidden_units_1=ARGS.hidden_units_1,
            hidden_units_2=ARGS.hidden_units_2,
        )
        model_embeddings = MLP(
            hidden_units_1=ARGS.hidden_units_1,
            hidden_units_2=ARGS.hidden_units_2,
        )
    elif ARGS.architecture == "rf":
        print(
            f"Comparing CLIP effectivity using two random forest models with {ARGS.n_estimators} trees..."
        )
        model_raw = RandomForest(
            n_estimators=ARGS.n_estimators,
            random_state=0,
            verbose=2,
        )
        model_embeddings = RandomForest(
            n_estimators=ARGS.n_estimators,
            random_state=0,
            verbose=2,
        )
    elif ARGS.architecture == "cnn":
        print(
            f"Comparing CLIP effectivity using two convolutional neural network models with {ARGS.conv1_filters} and {ARGS.conv2_filters} filters, {ARGS.fc1_units} fully connected units, and a {ARGS.kernel_size}x{ARGS.kernel_size} kernel..."
        )
        model_raw = CNN(
            conv1_filters=ARGS.conv1_filters,
            conv2_filters=ARGS.conv2_filters,
            fc1_units=ARGS.fc1_units,
            kernel_size=ARGS.kernel_size,
        )
        model_embeddings = CNN(
            conv1_filters=ARGS.conv1_filters,
            conv2_filters=ARGS.conv2_filters,
            fc1_units=ARGS.fc1_units,
            kernel_size=ARGS.kernel_size,
        )
    if ARGS.architecture == "mlp" or ARGS.architecture == "cnn":
        print(
            f"Training the models for {EPOCHS} epochs with a batch size of {ARGS.batch_size} and a learning rate of {ARGS.learning_rate}..."
        )

    accuracy_raw, test_accuracy_raw, train_cm_raw, test_cm_raw, losses_raw = (
        run_model_raw(model_raw)
    )
    (
        accuracy_embeddings,
        test_accuracy_embeddings,
        train_cm_embeddings,
        test_cm_embeddings,
        losses_embeddings,
    ) = run_model_embeddings(model_embeddings)

    results_path = os.path.join(
        os.getcwd(),
        "data",
        "results",
        time.strftime(f"{ARGS.architecture}_%m-%d_%I-%M%p"),
    )
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    cm_file = os.path.join(results_path, "confusion_matrices.npz")
    np.savez_compressed(
        cm_file,
        train_cm_raw=train_cm_raw,
        test_cm_raw=test_cm_raw,
        train_cm_embeddings=train_cm_embeddings,
        test_cm_embeddings=test_cm_embeddings,
    )
    print(f'Confusion matrices saved to "{cm_file}"')

    losses_file = os.path.join(results_path, "losses.npz")
    np.savez_compressed(
        losses_file, losses_raw=losses_raw, losses_embeddings=losses_embeddings
    )
    print(f'Losses saved to "{losses_file}"')

    hyperparams = {
        "architecture": ARGS.architecture,
    }

    if ARGS.architecture == "rf":
        hyperparams["n_estimators"] = ARGS.n_estimators
    elif ARGS.architecture == "mlp" or ARGS.architecture == "cnn":
        hyperparams["batch_size"] = ARGS.batch_size
        hyperparams["learning_rate"] = ARGS.learning_rate
        if ARGS.architecture == "mlp":
            hyperparams["hidden_units_1"] = ARGS.hidden_units_1
            hyperparams["hidden_units_2"] = ARGS.hidden_units_2
        elif ARGS.architecture == "cnn":
            hyperparams["conv1_filters"] = ARGS.conv1_filters
            hyperparams["conv2_filters"] = ARGS.conv2_filters
            hyperparams["fc1_units"] = ARGS.fc1_units
            hyperparams["kernel_size"] = ARGS.kernel_size
    hyperparams_file = os.path.join(results_path, "hyperparameters.json")
    with open(hyperparams_file, "w") as hyperparams_file:
        json.dump(hyperparams, hyperparams_file)
    print(f'Hyperparameters saved to "{hyperparams_file}"')

    accuracies = {
        "raw_train_accuracy": accuracy_raw,
        "raw_test_accuracy": test_accuracy_raw,
        "embedding_train_accuracy": accuracy_embeddings,
        "embedding_test_accuracy": test_accuracy_embeddings,
    }
    accuracies_file = os.path.join(results_path, "accuracies.json")
    with open(accuracies_file, "w") as results_file:
        json.dump(accuracies, results_file)
    print(f'Accuracies saved to "{accuracies_file}"')
