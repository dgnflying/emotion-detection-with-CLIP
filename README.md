
# emotion-detection

• Generates human facial emotion AI models

• Plots image embeddings and text embeddings for easy data collection and comparison

  

# Instructions
## Getting Started

1. Download the dataset at `https://www.kaggle.com/datasets/msambare/fer2013`

  

2. Unzip the file and add it to your workspace and rename the file as `faces`



3. Reconfigure your file to this ***EXACT*** structure:
   
```
	├── faces
	│   ├── raw_data
	│   │   ├── test
	|   │   │   ├── (images)
	│   │   ├── train
	|   │   │   ├── (images)
	│   ├── preprocessed_data
	│   │   ├── (empty)
 ```

5. Install project dependencies:

  

	```pip install scikit-learn```

	```pip install Pillow```

	```pip install tqdm```

	```pip install matplotlib```

## Creating a model

1. Run emotion_detection.py
2. Feel free to change your model's file name from `emotion_ai_#` to something more memorable

## Plotting embedding vectors
