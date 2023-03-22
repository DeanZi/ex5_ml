# ex5_ml
### Speech Command Classification using PyTorch with CNN

This is a PyTorch implementation of speech command classification using Convolutional Neural Networks (CNNs). 
The model used in this implementation is based on the VGG architecture.

### Requirements
* PyTorch
* NumPy

### Dataset
The Google Speech Commands Dataset was used to train and evaluate the model. 
The dataset consists of 30 short (1 second) audio clips, each of which corresponds to a different word.

### How to run the model

1.	Prerequisites:

•	On the same directory you should have:
o	My submitted files: ex5.py, gcommand_dataset.py
o	The data in a folder named gcommands

•	gcommands should be with the following content:
o	test
	file named as you wish
•	6836.wav
•	…
•	…
o	train 
	bed
	bird
	…
	…
o	valid
	bed
	bird
	…
	…

2.	After this, just run: `python3 ex5.py`
