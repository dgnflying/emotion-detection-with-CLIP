from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from torch.utils.data import TensorDataset, DataLoader

MODEL_ID = 'openai/clip-vit-base-patch16'

DATA_DIR = Path(__file__).parent / 'faces'
PREPROC_IMGS_DIR = DATA_DIR / 'preprocessed_data' / 'image_embeddings'
PREPROC_TEXT_DIR = DATA_DIR / 'preprocessed_data' / 'text_embeddings'
RAW_TRAIN_DIR = DATA_DIR / 'raw_data' / 'train'
RAW_TEST_DIR = DATA_DIR / 'raw_data' / 'test'
if not DATA_DIR.exists():
  raise ValueError(f'Path {str(DATA_DIR)} does not exist')
if not PREPROC_IMGS_DIR.exists():
  PREPROC_IMGS_DIR.mkdir(parents=True)
if not RAW_TRAIN_DIR.exists():
  raise ValueError(f'Path {str(RAW_TRAIN_DIR)} does not exist')
if not RAW_TEST_DIR.exists():
  raise ValueError(f'Path {str(RAW_TEST_DIR)} does not exist')
if not (set(p.name for p in RAW_TEST_DIR.iterdir() if p.is_dir()) <= set(p.name for p in RAW_TRAIN_DIR.iterdir() if p.is_dir())):
  raise ValueError('There are `test` labels that do not exist in `train`')

EMOTIONS = sorted((p.name for p in RAW_TRAIN_DIR.iterdir() if p.is_dir()))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_image_embeddings(
  directory,
  batch_size,
  model=None,
  processor=None
):

  if model == None:
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
  if processor == None:
    processor = AutoProcessor.from_pretrained(MODEL_ID)

  preproc_filename = PREPROC_IMGS_DIR / f'{directory.name}.npz'
  if preproc_filename.exists():
    print(f'Files for {directory.name} are already preprocessed.')
  else:
    imgs = np.stack([
      np.array(Image.open(filename).convert("RGB")).transpose(2, 0, 1)
      for emotion in tqdm(EMOTIONS, desc=f'Extracting {directory.name}ing data')
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
      batch_size=batch_size,
    )
    with torch.inference_mode():
      img_vecs = torch.cat([
        model.get_image_features(**processor(images=img_batch, return_tensors='pt').to(DEVICE))
        for (img_batch,) in tqdm(dataset, desc='Producing image vectors')
      ])
    img_vecs = np.array(img_vecs.cpu())
    np.savez_compressed(preproc_filename, vecs=img_vecs, targets=targets)
    print(f'Preprocessed data saved to "{preproc_filename}"')
    return img_vecs, targets

def create_text_embeddings(
  directory,
  model=None,
  processor=None
):

  if model == None:
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
  if processor == None:
    processor = AutoProcessor.from_pretrained(MODEL_ID)

  preproc_filename = PREPROC_TEXT_DIR / f'{directory.name}.npz'
  if preproc_filename.exists():
    print(f'Files for {directory.name} are already preprocessed.')
  else:
    phrases = []
    for emotion in EMOTIONS:
      if emotion == 'angry':
        phrases.append('An angry human face')
      elif emotion == 'disgust':
        phrases.append('A disgusted human face')
      elif emotion == 'fear':
        phrases.append('A fearful human face')
      elif emotion == 'happy':
        phrases.append('A happy human face')
      elif emotion == 'neutral':
        phrases.append('A neutral human face')
      elif emotion == 'sad':
        phrases.append('A sad human face')
      elif emotion == 'surprise':
        phrases.append('A surprised human face')
    targets = np.array(range(len(EMOTIONS)))
    with torch.inference_mode():
      text_vecs = model.get_text_features(**processor(text=phrases, return_tensors='pt', padding=True).to(DEVICE))

    text_vecs = np.array(text_vecs.cpu())
    np.savez_compressed(preproc_filename, vecs=text_vecs, targets=targets)
    print(f'Preprocessed data saved to "{preproc_filename}"')
    return text_vecs, targets