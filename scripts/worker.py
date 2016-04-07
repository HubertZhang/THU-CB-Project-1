import numpy as np
import theano
import theano.tensor as T
import lasagne
import time
import pickle
import os

import data_reader
import CONFIG


class Worker():
	def __init__(self, training_set, test_set):
		self.training_set = training_set
		self.test_set = test_set

		self.LAMBDA = 1e-4
		self.num_epochs = 10
		self.TrainingData = range(0, int(0.8*len(training_set)))
		self.ValidatingData = range(int(0.8*len(training_set)), len(training_set))
		self.BATCH_SIZE = 500

	def main(self):
		# Prepare Theano variables for inputs and targets
		input_var = T.tensor4('inputs')
		target_var = T.ivector('targets')

		network = self.build_cnn(input_var)
		prediction = lasagne.layers.get_output(network)
		loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
		loss = loss.mean() + self.LAMBDA * lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)

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
				print('Batch size: {}'.format(len(inputs)))
				train_err += train_fn(inputs, targets)
				train_batches += 1

	        # And a full pass over the validation data:
			val_err = 0
			val_acc = 0
			val_batches = 0
			for batch in self.training_batch(self.ValidatingData):
				inputs, targets = batch
				err, acc = val_fn(inputs, targets)
				val_err += err
				val_acc += acc
				val_batches += 1

      		# Then we print the results for this epoch:
			print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
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
			if 'inputs_{}.pkl'.format(index) in os.listdir('.'):
				print('Use saved file.')
				with open('inputs_{}.pkl'.format(index), 'rb') as f_in:
					inputs = pickle.load(f_in)
				with open('targets_{}.pkl'.format(index), 'rb') as f_in:
					targets = pickle.load(f_in)
				starting_pnt = 0
				while starting_pnt + self.BATCH_SIZE < len(inputs):
					yield np.array(inputs[starting_pnt:starting_pnt+self.BATCH_SIZE]).reshape(-1,1,CONFIG.AREA_SIZE,CONFIG.AREA_SIZE), np.array(targets[starting_pnt:starting_pnt+self.BATCH_SIZE],dtype='int32')
					starting_pnt += self.BATCH_SIZE
				
				yield np.array(inputs[starting_pnt:len(inputs)]).reshape(-1, 1, CONFIG.AREA_SIZE, CONFIG.AREA_SIZE), np.array(targets[starting_pnt:len(inputs)],dtype='int32')
				continue

			item = self.training_set[index]
			dataset = item.generate_training_set()
			print 'Length of dataset: {}'.format(len(dataset))
			inputs = []
			targets = []
			cnt_postive = 0
			cnt_negative = 0
			for pnt, flag in dataset:
				inputs.append(item.get_window(pnt))
				if flag:
				 	cnt_postive += 1
				 	targets.append(1)
				else:
					cnt_negative += 1
					targets.append(0)
				# targets.append(flag)
			print('Positive {}/ Negative {}'.format(cnt_postive, cnt_negative))
			with open('inputs_{}.pkl'.format(index), 'wb') as f_out:
				pickle.dump(inputs,f_out)
			with open('targets_{}.pkl'.format(index), 'wb') as f_out:
				pickle.dump(targets,f_out)

			starting_pnt = 0
			while starting_pnt + self.BATCH_SIZE < len(inputs):
				yield np.array(inputs[starting_pnt:starting_pnt+self.BATCH_SIZE]).reshape(-1,1,CONFIG.AREA_SIZE,CONFIG.AREA_SIZE), np.array(targets[starting_pnt:starting_pnt+self.BATCH_SIZE],dtype='int32')
				starting_pnt += self.BATCH_SIZE
				
			yield np.array(inputs[starting_pnt:len(inputs)]).reshape(-1, 1, CONFIG.AREA_SIZE, CONFIG.AREA_SIZE), np.array(targets[starting_pnt:len(inputs)],dtype='int32')
			continue


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
				network, num_filters=32, filter_size=(5, 5),
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
		network = lasagne.layers.Conv2DLayer(
				network, num_filters=32, filter_size=(5, 5),
				nonlinearity=lasagne.nonlinearities.rectify)
		network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

	    # A fully-connected layer of 256 units with 50% dropout on its inputs:
		network = lasagne.layers.DenseLayer(
				lasagne.layers.dropout(network, p=.5),
				num_units=256,
				nonlinearity=lasagne.nonlinearities.rectify)

    	# And, finally, the 10-unit output layer with 50% dropout on its inputs:
		network = lasagne.layers.DenseLayer(
				lasagne.layers.dropout(network, p=.5),
				num_units=2,
				nonlinearity=lasagne.nonlinearities.softmax)

		return network

