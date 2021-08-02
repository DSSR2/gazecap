# Gaze Capture
## About
This is a re-implementation of [Eye Tracking for Everyone](https://gazecapture.csail.mit.edu/) in PyTorch. 
Some code is derived from the  [official git repository](https://github.com/CSAILVision/GazeCapture). 

This implementation significantly reduces training time by: 
* Improved data organization 
* Multi GPU support with PyTorch Lightning

Also added comet.ml support to track experiments live as the model trains. 

The model file provided here is trained on only iPhone data and no augmentations.

## Dataset details
The dataset is available on registration from [here](https://gazecapture.csail.mit.edu/download.php)

The raw dataset is HUGE ~135GB of data. Use the [prepareDataset](Utils/prepareDataset.py) script to convert the dataset to a more usable (and smaller) version. 

Models provided are trained on only iPhone data. 
Total number of files: 1,272,185
Number of train files: 1,076,797
Number of validation files: 51,592
Number of test files: 143,796