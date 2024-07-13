
# clip-capabilities

•	Generate AI models that detect human facial emotion and text/image embeddings of human facial-related data

•	Display emotion model performance data (i.e. loss curve, confusion matrix, etc.)

•	Plot image and text embeddings of human facial-related data for collection and comparison

  

# Instructions
## Dependencies

Install with **pip**:
	
	```pip install scikit-learn```
	
	```pip install Pillow```
	
	```pip install tqdm```
	
	```pip install matplotlib```
	
	```pip install seaborn```

Or install with **conda**:

	```conda install scikit-learn```
	
	```conda install pillow```
	
	```conda install tqdm```
	
	```conda install matplotlib```
	
	```conda install seaborn```

## Creating a model

1. Run `py generate_model.py` in the terminal:

	### Arguments:
	
	#### `--hidden_layers` **OR** `-l`:
	The hidden layers of the model ***(default: 100)***\
	**Example:** `py generate_model.py -l 1000 500 100` generates a model architecture of `input -> 1000 nodes -> 500 nodes -> 100 nodes -> output`
	
	#### `--batch_size` **OR** `-b`:
	The batch size for training the model ***(default: 64)***

	#### `--learning_rate` **OR** `-r`:
	The learning rate of the model ***(default: 0.001)***

	#### `--max_iter` **OR** `-m`:
	The maximum number of iterations for training the model ***(default: 200)***

3. Once finished, feel free to change your model's folder name at `/output/*current_date*-*iteration_number*` to something more memorable
   
   ***WARNING: If the original file name is changed from*** `*current_date*-*iteration_number*`, ***the*** `--use_date` ***(***`-d`***)*** ***argument within*** `plot_data.py` ***will be unusable***

## Replotting past model data

1. Run `py plot_data.py` in the terminal

	### Arguments:
	#### `--file` **OR** `-f`:
 	The file containing the model data\
	**Example:** `py plot_data.py -f 2099-04-01-0` will replot the data of the user's first generated model from April 1st, 2099 (if the file has not been renamed)

	#### `--use_date` **OR** `-d`:
	Use the current date at the beginning of file specification ***(NOT TO BE USED WHEN THE TARGET FILE HAS BEEN RENAMED)***\
	**Example:** `py plot_data.py -d -f 0` will replot the data of the user's first generated model from the day this command was ran (if the file has not been renamed)
	
	#### `--confusion_matrix` **OR**  `-c`:
	Display the model's confusion matrices
	
	#### `--loss_curve` **OR** `-l`:
 	Display the model's loss curve

## Plotting CLIP embedding vectors

1. Run `py plot_embeddings.py` in the terminal:

   	### Arguments:
	#### `--batch_size` **OR** `-b`:
	Batch size to feed encoder to produce vector embeddings ***(default: 32)***
	
	#### `--average` **OR** `-a`:
	Display the average vector for each emotion
	
	#### `--all` **OR** `-A`:
	Display all embeddings for every emotion

	#### `--text` **OR** `-t`:
	Display the text embeddings of the emotions

	#### `--comparison` **OR** `-c`:
	Display comparisons between each emotion's average image vector and their text counterpart

	#### `--titles` **OR** `-T`:
	Generate titles for the plots
