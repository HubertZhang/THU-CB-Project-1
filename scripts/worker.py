import numpy as np
import theano
import theano.tensor as T
import lasagne
import time
import pickle
import os
from scipy.ndimage.filters import gaussian_filter

import data_reader
import CONFIG


class Worker():
	def __init__(self, training_set, test_set):
		self.training_set = training_set
		self.test_set = test_set
		for index in range(len(self.training_set)):
			# self.training_set[index].image_data = gaussian_filter(self.training_set[index].image_data, 15)
			self.training_set[index].image_data -= np.min(self.training_set[index].image_data)
			self.training_set[index].image_data /= np.max(self.training_set[index].image_data)
		for index in range(len(self.test_set)):
			# self.test_set[index].image_data = gaussian_filter(self.test_set[index].image_data, 15)
			self.test_set[index].image_data -= np.min(self.test_set[index].image_data)
			self.test_set[index].image_data /= np.max(self.test_set[index].image_data)

		self.LAMBDA = 1e-4
		self.num_epochs = 100
		self.TrainingData = range(0, int(0.8*len(training_set)))
		self.ValidatingData = range(int(0.8*len(training_set)), len(training_set))
		# self.TrainingData = range(0, 10)
		# self.ValidatingData = range(len(training_set)-6, len(training_set))
		self.BATCH_SIZE = 500

	def main(self):
		print("WINDOW SIZE: {}".format(CONFIG.AREA_SIZE))
		# Prepare Theano variables for inputs and targets
		input_var = T.tensor4('inputs')
		target_var = T.ivector('targets')

		network = self.build_cnn(input_var)
		prediction = lasagne.layers.get_output(network)
		loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
		loss = loss.mean()#  + self.LAMBDA * lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)

	    # Create update expressions for training, i.e., how to modify the
	    # parameters at each training step. Here, we'll use Stochastic Gradient
    	# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
		params = lasagne.layers.get_all_params(network, trainable=True)
		updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=CONFIG.LEARNING_RATE, momentum=CONFIG.MOMENTUM)

	    # Create a loss expression for validation/testing. The crucial difference
    	# here is that we do a deterministic forward pass through the network,
    	# disabling dropout layers.
		test_prediction = lasagne.layers.get_output(network, deterministic=True)
		test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
		test_loss = test_loss.mean()
    	# As a bonus, also create an expression for the classification accuracy:
		test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
	                      dtype=theano.config.floatX)

    	# Compile a function performing a training step on a mini-batch (by giving
	    # the updates dictionary) and returning the corresponding training loss:
		train_fn = theano.function([input_var, target_var], loss, updates=updates)

		result_fn = theano.function([input_var], test_prediction)
		structure_fn = theano.function([], lasagne.layers.get_all_params(network))

	    # Compile a second function computing the validation loss and accuracy:
		val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

	    # Finally, launch the training loop.
		print("Starting training...")
	    # We iterate over epochs:
		for epoch in range(self.num_epochs):
        	# In each epoch, we do a full pass over the training data:
			train_err = 0
			train_batches = 0
			start_time = time.time()
			for batch in self.training_batch(self.TrainingData):
				inputs, targets = batch
				# print('Batch size: {}'.format(len(inputs)))
				# print inputs
				# print targets
				train_err += train_fn(inputs, targets) * len(inputs)
				# print(result_fn(inputs))
				structure = structure_fn()
				# for index, layer in enumerate(structure):
				# 	print('Layer {}'.format(index))
				# 	print('Shape: {}'.format(layer.shape))
				# 	print(layer)
				train_batches += len(inputs)
				# print('Training loss: {}'.format(train_err / train_batches))

	        # And a full pass over the validation data:
			val_err = 0
			val_acc = 0
			val_batches = 0
			for batch in self.training_batch(self.ValidatingData):
				inputs, targets = batch
				err, acc = val_fn(inputs, targets)
				# print(result_fn(inputs))
				val_err += err * len(inputs)
				val_acc += acc * len(inputs)
				val_batches += len(inputs)

      		# Then we print the results for this epoch:
			print("Epoch {} of {} took {:.3f}s".format(epoch + 1, self.num_epochs, time.time() - start_time))
			print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
			print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
			print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

	    # After training, we compute and print the test error:
     	# test_err = 0
	    # test_acc = 0
    	# test_batches = 0
	    # for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
    	#     inputs, targets = batch
        # 	err, acc = val_fn(inputs, targets)
	    #     test_err += err
    	#     test_acc += acc
        # 	test_batches += 1
	    # print("Final results:")
    	# print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
	    # print("  test accuracy:\t\t{:.2f} %".format(
    	#     test_acc / test_batches * 100))

	    # Optionally, you could now dump the network weights to a file like this:
    	# np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
	    #
    	# And load them again later on like this:
	    # with np.load('model.npz') as f:
    	#     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	    # lasagne.layers.set_all_param_values(network, param_values)

	def training_batch(self, data_slice):
		for index in data_slice:
			# if 'inputs_{}.pkl'.format(index) in os.listdir('.'):
			# 	# print('Use saved file.')
			# 	with open('inputs_{}.pkl'.format(index), 'rb') as f_in:
			# 		inputs = pickle.load(f_in)
			# 	with open('targets_{}.pkl'.format(index), 'rb') as f_in:
			# 		targets = pickle.load(f_in)
			# 	starting_pnt = 0
			# 	while starting_pnt + self.BATCH_SIZE < len(inputs):
			# 		yield np.array(inputs[starting_pnt:starting_pnt+self.BATCH_SIZE]).reshape(-1,1,CONFIG.AREA_SIZE,CONFIG.AREA_SIZE), np.array(targets[starting_pnt:starting_pnt+self.BATCH_SIZE],dtype='int32')
			# 		starting_pnt += self.BATCH_SIZE
			# 	
			# 	yield np.array(inputs[starting_pnt:len(inputs)]).reshape(-1, 1, CONFIG.AREA_SIZE, CONFIG.AREA_SIZE), np.array(targets[starting_pnt:len(inputs)],dtype='int32')
			# 	continue

			item = self.training_set[index]
			dataset = item.generate_training_set()
			# print 'Length of dataset: {}'.format(len(dataset))
			inputs = []
			targets = []
			cnt_postive = 0
			cnt_negative = 0
			start_time = time.time()
			for pnt, flag in dataset:
				inputs.append(item.get_window(pnt))
				if flag:
					cnt_postive += 1
					targets.append(1)
				else:
					cnt_negative += 1
					targets.append(0)
				# targets.append(flag)
			# print('Loading feature took {:.3f}s'.format(time.time()-start_time))
			# print('Positive {}/ Negative {}'.format(cnt_postive, cnt_negative))
			# with open('inputs_{}.pkl'.format(index), 'wb') as f_out:
			# 	pickle.dump(inputs,f_out)
			# with open('targets_{}.pkl'.format(index), 'wb') as f_out:
			# 	pickle.dump(targets,f_out)

			starting_pnt = 0
			while starting_pnt + self.BATCH_SIZE < len(inputs):
				yield np.array(inputs[starting_pnt:starting_pnt+self.BATCH_SIZE]).reshape(-1,1,CONFIG.AREA_SIZE,CONFIG.AREA_SIZE), np.array(targets[starting_pnt:starting_pnt+self.BATCH_SIZE],dtype='int32')
				starting_pnt += self.BATCH_SIZE
				
			yield np.array(inputs[starting_pnt:len(inputs)]).reshape(-1, 1, CONFIG.AREA_SIZE, CONFIG.AREA_SIZE), np.array(targets[starting_pnt:len(inputs)],dtype='int32')
			continue

	def main_minst(self):
		X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

		# Prepare Theano variables for inputs and targets
		input_var = T.tensor4('inputs')
		target_var = T.ivector('targets')

		network = self.build_cnn_mnist(input_var)
		prediction = lasagne.layers.get_output(network)
		loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
		loss = loss.mean()#  + self.LAMBDA * lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)

	    # Create update expressions for training, i.e., how to modify the
	    # parameters at each training step. Here, we'll use Stochastic Gradient
    	# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
		params = lasagne.layers.get_all_params(network, trainable=True)
		updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

	    # Create a loss expression for validation/testing. The crucial difference
    	# here is that we do a deterministic forward pass through the network,
    	# disabling dropout layers.
		test_prediction = lasagne.layers.get_output(network, deterministic=True)
		test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
		test_loss = test_loss.mean()
    	# As a bonus, also create an expression for the classification accuracy:
		test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
	                      dtype=theano.config.floatX)

    	# Compile a function performing a training step on a mini-batch (by giving
	    # the updates dictionary) and returning the corresponding training loss:
		train_fn = theano.function([input_var, target_var], loss, updates=updates)

		result_fn = theano.function([input_var], test_prediction)
		structure_fn = theano.function([], lasagne.layers.get_all_params(network))

	    # Compile a second function computing the validation loss and accuracy:
		val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

	    # Finally, launch the training loop.
		print("Starting training...")
	    # We iterate over epochs:
		for epoch in range(self.num_epochs):
			if epoch % 10 == 9:
				np.savez('model_{}.npz'.format(epoch/10), *lasagne.layers.get_all_param_values(network))
        	# In each epoch, we do a full pass over the training data:
			train_err = 0
			train_batches = 0
			start_time = time.time()
			for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
				inputs, targets = batch
				# print('Batch size: {}'.format(len(inputs)))
				# for row in inputs:
				# 	print(row)
				# print targets
				train_err += train_fn(inputs, targets) * len(inputs)
				# print(result_fn(inputs))
				structure = structure_fn()
				# print(inputs.shape)
				# for index, layer in enumerate(structure):
				# 	print('Layer {}'.format(index))
				# 	print('Shape: {}'.format(layer.shape))
				# 	print(layer)
				train_batches += len(inputs)
				# print('Training loss: {}'.format(train_err / train_batches))

	        # And a full pass over the validation data:
			val_err = 0
			val_acc = 0
			val_batches = 0
			for batch in iterate_minibatches(X_val, y_val, 500):
				inputs, targets = batch
				err, acc = val_fn(inputs, targets)
				# print(result_fn(inputs))
				val_err += err * len(inputs)
				val_acc += acc * len(inputs)
				val_batches += len(inputs)

      		# Then we print the results for this epoch:
			print("Epoch {} of {} took {:.3f}s".format(epoch + 1, self.num_epochs, time.time() - start_time))
			print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
			print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
			print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

	def build_cnn(self, input_var=None):
	    # As a third model, we'll create a CNN of two convolution + pooling stages
    	# and a fully-connected hidden layer in front of the output layer.

	    # Input layer, as usual:
		network = lasagne.layers.InputLayer(shape=(None, 1, CONFIG.AREA_SIZE, CONFIG.AREA_SIZE),
        	                                input_var=input_var)
	    # This time we do not apply input dropout, as it tends to work less well
    	# for convolutional layers.

	    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    	# convolutions are supported as well; see the docstring.
		network = lasagne.layers.Conv2DLayer(
				network, num_filters=96, filter_size=(11, 11), stride=(4,4),
				nonlinearity=lasagne.nonlinearities.rectify,
				W=lasagne.init.GlorotUniform())
	    # Expert note: Lasagne provides alternative convolutional layers that
    	# override Theano's choice of which implementation to use; for details
    	# please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

	    # Max-pooling layer of factor 2 in both dimensions:
		network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

	    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
		network = lasagne.layers.Conv2DLayer(
				network, num_filters=256, filter_size=(5, 5),
				nonlinearity=lasagne.nonlinearities.rectify)
		network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

	    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
		network = lasagne.layers.Conv2DLayer(
				network, num_filters=384, filter_size=(3, 3),
				nonlinearity=lasagne.nonlinearities.rectify)
		network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

	    #  A fully-connected layer of 256 units with 50% dropout on its inputs:
		network = lasagne.layers.DenseLayer(
				lasagne.layers.dropout(network, p=.5),
				num_units=1024,
				nonlinearity=lasagne.nonlinearities.rectify)

    	# And, finally, the 10-unit output layer with 50% dropout on its inputs:
		network = lasagne.layers.DenseLayer(
				lasagne.layers.dropout(network, p=.5),
				num_units=2,
				nonlinearity=lasagne.nonlinearities.softmax)

		return network

	def build_cnn_mnist(self, input_var=None):
	    # As a third model, we'll create a CNN of two convolution + pooling stages
    	# and a fully-connected hidden layer in front of the output layer.

	    # Input layer, as usual:
		network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
        	                                input_var=input_var)
	    # This time we do not apply input dropout, as it tends to work less well
    	# for convolutional layers.

	    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    	# convolutions are supported as well; see the docstring.
		network = lasagne.layers.Conv2DLayer(
				network, num_filters=32, filter_size=(5,5),
				nonlinearity=lasagne.nonlinearities.rectify,
				W=lasagne.init.GlorotUniform())
	    # Expert note: Lasagne provides alternative convolutional layers that
    	# override Theano's choice of which implementation to use; for details
    	# please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

	    # Max-pooling layer of factor 2 in both dimensions:
		network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

	    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
		network = lasagne.layers.Conv2DLayer(
				network, num_filters=32, filter_size=(5, 5),
				nonlinearity=lasagne.nonlinearities.rectify)
		network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

	    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
		# network = lasagne.layers.Conv2DLayer(
		# 		network, num_filters=384, filter_size=(3, 3),
		# 		nonlinearity=lasagne.nonlinearities.rectify)
		# network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

	    #  A fully-connected layer of 256 units with 50% dropout on its inputs:
		network = lasagne.layers.DenseLayer(
				lasagne.layers.dropout(network, p=.5),
				num_units=256,
				nonlinearity=lasagne.nonlinearities.rectify)

    	# And, finally, the 10-unit output layer with 50% dropout on its inputs:
		network = lasagne.layers.DenseLayer(
				lasagne.layers.dropout(network, p=.5),
				num_units=10,
				nonlinearity=lasagne.nonlinearities.softmax)

		return network

import sys
import os
import time
def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
