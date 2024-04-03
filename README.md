
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
	
	#### `--hidden_layers` **OR** `-l`:
	The hidden layers of the model ***(default: 100)***\
	**Example:** `py generate_model.py -l 1000 100` generates a model architecture of `input -> 1000 nodes -> 100 nodes -> output`
	
	#### `--batch_size` **OR** `-b`:
	Opt out of showing comparisons between each emotion's average image vector and their text counterpart ***(default: 200)***

	#### `--learning_rate` **OR** `-r`:
	The learning rate of the model ***(default: 0.001)***

2. Once finished, if you saved your model by omitting `--no_save` as an argument, feel free to change your model's folder name at `/output/*current_date*-*iteration_number*` to something more memorable (just note that the `--use_current_date` (`-d`) argument within `replot_data.py` will be unusable)

## Replotting past model data

1. Run `py replot_data.py` in the terminal

	### Arguments:
	#### `--file` **OR** `-f`:
 	The file containing the model data\
	**Example:** `py replot_data.py -f 2099-04-01-0` will replot the data of the user's first generated model from April 1st, 2099 (if the file was not renamed)

	#### `--use_current_date` **OR** `-d`:
	Use the current data as the first three values in file specification ***(NOT TO BE USED WHEN THE TARGET FILE HAS BEEN RENAMED)***\
	**Example:** `py replot_data.py -d -f 0` will replot the data of the user's first generated model from the day this was ran (if the file was not renamed)
	
	#### `--no_cm` **OR**  `-c`:
	Opt out of displaying the model's confusion matrices
	
	#### `--no_loss_curve` **OR** `-l`:
 	Opt out of displaying the model's loss curve

## Plotting CLIP embedding vectors

1. Run `py plot_embeddings.py` in the terminal:

   	### Arguments:
	#### `--batch_size` **OR** `-b`:
	Batch size to feed encoder to produce vector embeddings ***(default: 32)***
	
	#### `--no_average` **OR** `-a`:
	Opt out of showing the average vector for each emotion
	
	#### `--no_all` **OR** `-A`:
	Opt out of showing all embeddings for every emotion

	#### `--no_comparison` **OR** `-c`:
	Opt out of showing comparisons between each emotion's average image vector and their text counterpart
