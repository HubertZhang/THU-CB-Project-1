#Partical Picker
##Requirement

	Python
	numpy
	Theano
	lagasne

##Run the picker
Data file should be stored in `data` folder while test data in `data/test` and training data in `data/training`

Please put `model.npz` in the `output` folder then run at `scripts`
	
	python main.py ../data ../output


This script would run detector on the last one test data, and return the detected particals' position, precision and recall.

##Train the model
To train the model, please uncomment following line 

	alg.train_model()
in the `main.py` and comment following line

	alg.load_model('model.npz')

##Misc
In class `Worker`, following functions is useful

`train_model` would train the whole network according to the parameters in `CONFIG.py`

`predict` accepts a `DataItem` object and would predict the particals' position, and its precision and recall.

`load_model` is used to load a existed model for predicting or further training.
