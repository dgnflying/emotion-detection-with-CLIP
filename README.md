
# emotion-detection

•	Generates AI models that detect human facial emotion

•	Plots text and image embeddings of human facial-related data for collection and comparison

  

# Instructions
## Getting Started

1. Download the faces dataset at `https://www.kaggle.com/datasets/msambare/fer2013`

  

2. Unzip the file and add it to your workspace as `faces`



3. Reconfigure your `faces` to this ***EXACT*** structure:
   
```
	├── faces
	│   ├── raw_data
	│   │   ├── test
	|   │   │   ├── (images)
	│   │   ├── train
	|   │   │   ├── (images)
	│   ├── preprocessed_data
	│   │   ├── image_embeddings
	|   │   │   ├── (empty)
	│   │   ├── text_embeddings
	|   │   │   ├── (empty)
 ```

5. Install project dependencies:

	With **pip**:
	
	```pip install scikit-learn```
	
	```pip install Pillow```
	
	```pip install tqdm```
	
	```pip install matplotlib```
	
	Or with **conda**:
	
	```conda install scikit-learn```
	
	```conda install pillow```
	
	```conda install tqdm```
	
	```conda install matplotlib```

## Creating a model

1. Run `py generate_model.py` in the terminal:

	### Arguments:
	#### `--no_save` **OR** `-s`:
	Opt out of saving the model
	
	#### `--hidden_layers` **OR** `-h`:
	The hidden layers of the model ***(default: 100)***\
	**Example:** `py generate_model.py -h 1000 100` generates a model architecture of `input -> 1000 nodes -> 100 nodes -> output`
	
	#### `--batch_size` **OR** `-b`:
	Opt out of showing comparisons between each emotion's average image vector and their text counterpart ***(default: 200)***


3. Once finished, if you saved your model by omitting `--no_save` as an argument, feel free to change your model's file name in `root/models` from `emotion_ai_#` to something more memorable

## Plotting embedding vectors

1. Run `py plot_embeddings.py` in the terminal

	### Arguments:
	#### `--no_average` **OR** `-a`:
	Opt out of showing the average vector for each emotion
	
	#### `--no_all` **OR**  `-A`:
	Opt out of showing all embeddings for every emotion
	
	#### `--no_comparison` **OR** `-c`:
 	Opt out of showing comparisons between each emotion's average image vector and their text counterpart

	#### `--batch_size` **OR** `-b`:
 	Batch size to feed encoder to produce vector embeddings ***(default: 32)***
