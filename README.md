
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

	#### Arguments:
	•	`-s` **OR** `--no_save`: Opt in to saving the model
	
	•	`-h` **OR** `--hidden_layers`: The hidden layers of the model ***(default: 100)***\
		##### Example:
		`py generate_model.py -h 1000 100` generates a model architecture of `input -> 1000 nodes -> 100 nodes -> output`
	
	•	`-b` **OR** `--batch_size`: Opt out of showing comparisons between each emotion's average image vector and their text counterpart ***(default: 200)***


3. Once finished, if you saved your model by omitting `--no_save` as an argument, feel free to change your model's file name in `root/models` from `emotion_ai_#` to something more memorable

## Plotting embedding vectors

1. Run `py plot_embeddings.py` in the terminal

	#### Arguments:
	•	`-a` **OR** `--no_average`: Opt out of showing the average vector for each emotion
	
	•	`-A` **OR** `--no_all`: Opt out of showing all embeddings for every emotion
	
	•	`-c` **OR** `--no_comparison`: Opt out of showing comparisons between each emotion's average image vector and their text counterpart

	•	`-b` **OR** `--batch_size`: Batch size to feed encoder to produce vector embeddings ***(default: 32)***
