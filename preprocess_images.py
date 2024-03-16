from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from torch.utils.data import TensorDataset, DataLoader

MODEL_ID = 'openai/clip-vit-base-patch16'

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

def preprocess_images(
    directory,
    batch_size,
    model=AutoModel.from_pretrained(MODEL_ID),
    processor=AutoProcessor.from_pretrained(MODEL_ID)
  ):

  preproc_filename = DATA_DIR / "preprocessed_data" / f'preprocessed_{directory.name}_data.npz'
  if preproc_filename.exists():
    print(f'Files for {directory.name} are already preprocessed.')
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
      batch_size=batch_size,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    with torch.inference_mode():
      img_vecs = torch.cat([
        model.get_image_features(**processor(images=img_batch, return_tensors='pt').to(device))
        for (img_batch,) in tqdm(dataset, desc='Producing image vectors')
      ])
    img_vecs = np.array(img_vecs.cpu())
    Path("faces/preprocessed_data").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(preproc_filename, img_vecs=img_vecs, targets=targets)
    print(f'Preprocessed data saved to "{preproc_filename}"')
    return img_vecs, targets