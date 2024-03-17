
# emotion-detection

•	Generates AI models that detect human facial emotion

•	Plots text and image embeddings of human facial-related data for collection and comparison

  

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
