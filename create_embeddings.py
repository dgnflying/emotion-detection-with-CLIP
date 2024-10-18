import os
import argparse

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from torchvision import transforms, datasets

parser = argparse.ArgumentParser(
    description="Generate embeddings for images and text using OpenAI's CLIP model"
)

parser.add_argument(
    "--batch_size",
    "-b",
    type=int,
    default=32,
    help="The batch size for generating embeddings",
)

parser.add_argument(
    "--create_image_embeddings",
    "-i",
    help="Create image embeddings",
    action=argparse.BooleanOptionalAction,
)

parser.add_argument(
    "--create_text_embeddings",
    "-t",
    help="Create text embeddings",
    action=argparse.BooleanOptionalAction,
)

ARGS = parser.parse_args()

MODEL_ID = "openai/clip-vit-base-patch16"

data_path = os.path.join(os.getcwd(), "data")
embeddings_path = os.path.join(data_path, "fer2013", "embedding_data")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"


def custom_transform(img):
    return np.array(img.convert("RGB")).transpose(2, 0, 1)


def create_image_embeddings(
    data_path=data_path,
    batch_size=ARGS.batch_size,
    model=None,
    processor=None,
):

    if model == None:
        model = AutoModel.from_pretrained(MODEL_ID).to(device)
    if processor == None:
        processor = AutoProcessor.from_pretrained(MODEL_ID)

    train_embeddings_file = os.path.join(embeddings_path, "train_image_embeddings.npz")
    test_embeddings_file = os.path.join(embeddings_path, "test_image_embeddings.npz")

    # Ensure the embedding directory exists
    os.makedirs(embeddings_path, exist_ok=True)

    if os.path.exists(train_embeddings_file):
        print("Training image embeddings found")
    else:
        transform = transforms.Compose([transforms.Lambda(custom_transform)])
        train_set = datasets.FER2013(root=data_path, transform=transform, split="train")

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=8
        )
        with torch.inference_mode():
            train_img_embeddings = []
            train_labels = []
            for image_batch, label_batch in tqdm(train_loader, desc="Producing train image vectors"):
                features = model.get_image_features(
                    **processor(images=image_batch, return_tensors="pt").to(device)
                )
                train_labels.append(label_batch)
                train_img_embeddings.append(features)
            train_img_embeddings = torch.cat(train_img_embeddings)
            train_labels = torch.cat(train_labels)

        train_img_embeddings = np.array(train_img_embeddings.cpu())
        train_labels = np.array(train_labels.cpu())
        np.savez_compressed(train_embeddings_file, vecs=train_img_embeddings, targets=train_labels)
        print(f'Training image embeddings saved to "{train_embeddings_file}"')

    if os.path.exists(test_embeddings_file):
        print("Testing image embeddings found")
    else:
        transform = transforms.Compose([transforms.Lambda(custom_transform)])
        test_set = datasets.FER2013(root=data_path, transform=transform, split="train")

        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=True, num_workers=8
        )
        with torch.inference_mode():
            test_img_embeddings = []
            test_labels = []
            for image_batch, label_batch in tqdm(test_loader, desc="Producing test image vectors"):
                features = model.get_image_features(
                    **processor(images=image_batch, return_tensors="pt").to(device)
                )
                test_labels.append(label_batch)
                test_img_embeddings.append(features)
            test_img_embeddings = torch.cat(test_img_embeddings)
            test_labels = torch.cat(test_labels)

        test_img_embeddings = np.array(test_img_embeddings.cpu())
        test_labels = np.array(test_labels.cpu())
        np.savez_compressed(test_embeddings_file, vecs=test_img_embeddings, targets=test_labels)
        print(f'Testing image embeddings saved to "{test_embeddings_file}"')
        return train_img_embeddings, train_labels, test_img_embeddings, test_labels


def create_text_embeddings(model=None, processor=None):

    if model == None:
        model = AutoModel.from_pretrained(MODEL_ID).to(device)
    if processor == None:
        processor = AutoProcessor.from_pretrained(MODEL_ID)

    text_embeddings_file = os.path.join(embeddings_path, "text_embeddings.npz")

    # Ensure the embedding directory exists
    os.makedirs(embeddings_path, exist_ok=True)

    if os.path.exists(text_embeddings_file):
        print("Text embeddings found")
    else:
        phrases = [
            "An angry human face",
            "A disgusted human face",
            "A fearful human face",
            "A happy human face",
            "A neutral human face",
            "A sad human face",
            "A surprised human face",
        ]
        targets = np.array(range(len(phrases)))
        with torch.inference_mode():
            text_vecs = model.get_text_features(
                **processor(text=phrases, return_tensors="pt", padding=True).to(device)
            )

        text_vecs = np.array(text_vecs.cpu())
        np.savez_compressed(text_embeddings_file, vecs=text_vecs, targets=targets)
        print(f'Preprocessed data saved to "{text_embeddings_file}"')
        return text_vecs, targets


if __name__ == "__main__":
    if ARGS.create_image_embeddings:
        create_image_embeddings()
    if ARGS.create_text_embeddings:
        create_text_embeddings()
