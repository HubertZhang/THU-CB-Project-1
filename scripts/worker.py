import numpy as np
import theano
import theano.tensor as T
import lasagne
import time
import pickle
import os
from scipy.ndimage.filters import gaussian_filter
import logging
import matplotlib.pyplot as plt

import data_reader
import CONFIG


class Worker():
	def __init__(self, training_set, test_set):
		self.training_set = training_set
		# for item in self.training_set:
		# 	item.extend_image()
		self.test_set = test_set

		for index in range(len(self.training_set)):
			# self.training_set[index].image_data = gaussian_filter(self.training_set[index].image_data, 15)
			self.training_set[index].image_data -= np.min(self.training_set[index].image_data)
			self.training_set[index].image_data /= np.max(self.training_set[index].image_data)
		for index in range(len(self.test_set)):
			# self.test_set[index].image_data = gaussian_filter(self.test_set[index].image_data, 15)
			self.test_set[index].image_data -= np.min(self.test_set[index].image_data)
			self.test_set[index].image_data /= np.max(self.test_set[index].image_data)

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

		self.network = self.build_cnn(input_var)
		prediction = lasagne.layers.get_output(self.network)
		loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
		loss = loss.mean()#  + self.LAMBDA * lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)

	    # Create update expressions for training, i.e., how to modify the
	    # parameters at each training step. Here, we'll use Stochastic Gradient
    	# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
		params = lasagne.layers.get_all_params(self.network, trainable=True)
		# updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=CONFIG.LEARNING_RATE, momentum=CONFIG.MOMENTUM)
		updates = lasagne.updates.adadelta(loss, params, learning_rate=CONFIG.LEARNING_RATE)

	    # Create a loss expression for validation/testing. The crucial difference
    	# here is that we do a deterministic forward pass through the network,
    	# disabling dropout layers.
		test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
		test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
		test_loss = test_loss.mean()
    	# As a bonus, also create an expression for the classification accuracy:
		test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
	                      dtype=theano.config.floatX)

    	# Compile a function performing a training step on a mini-batch (by giving
	    # the updates dictionary) and returning the corresponding training loss:
		train_fn = theano.function([input_var, target_var], loss, updates=updates)

		result_fn = theano.function([input_var], test_prediction)
		structure_fn = theano.function([], lasagne.layers.get_all_params(self.network))

	    # Compile a second function computing the validation loss and accuracy:
		val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

		# accuracy = 0.
		# diff_accuracy = 0.
	    # Finally, launch the training loop.
		print("Starting training...")
	    # We iterate over epochs:
		for epoch in range(self.num_epochs):
			# if epoch in [50, 75, 90]:
			# 	print("Update Learning Rate.")
			# 	logging.info('Update Learning Rate.')
			# 	CONFIG.LEARNING_RATE = CONFIG.LEARNING_RATE / 2.
			# 	CONFIG.MOMENTUM = CONFIG.MOMENTUM / 2.
			# 	updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=CONFIG.LEARNING_RATE, momentum=CONFIG.MOMENTUM)
			# 	train_fn = theano.function([input_var, target_var], loss, updates=updates)

			if epoch % 10 == 9:
				np.savez('model_{}.npz'.format(epoch/10), *lasagne.layers.get_all_param_values(self.network))
        	# In each epoch, we do a full pass over the training data:
			train_err = 0
			train_batches = 0
			start_time = time.time()
			for batch in self.training_batch(self.TrainingData):
				inputs, targets = batch
				print('Batch size: {}'.format(len(inputs)))
				# print inputs
				# print targets
				train_err += train_fn(inputs, targets) * len(inputs)
				# print(result_fn(inputs))
				# structure = structure_fn()
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

			# diff_accuracy = val_acc / val_batches - accuracy
			# accuracy = val_acc / val_batches
      		# Then we print the results for this epoch:
			print("Epoch {} of {} took {:.3f}s".format(epoch + 1, self.num_epochs, time.time() - start_time))
			logging.info("Epoch {} of {} took {:.3f}s".format(epoch + 1, self.num_epochs, time.time() - start_time))
			print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
			logging.info("  training loss:\t\t{:.6f}".format(train_err / train_batches))
			print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
			logging.info("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
			print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))
			logging.info("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

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

	def predict(self, img, model_name):
		img.extend_image()

		input_var = T.tensor4('inputs')
		target_var = T.ivector('targets')

		self.network = self.build_cnn(input_var)
		with np.load(model_name) as f:
		    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
		lasagne.layers.set_all_param_values(self.network, param_values)

		input_pnts = []
		mat_dim = 0
		for i in range(CONFIG.HALF_AREA_SIZE, img.img_dim[0]-CONFIG.HALF_AREA_SIZE, CONFIG.STEP):
		# for i in range(1000, 1500, CONFIG.STEP):
			mat_dim += 1
			for j in range(CONFIG.HALF_AREA_SIZE, img.img_dim[1]-CONFIG.HALF_AREA_SIZE, CONFIG.STEP):
			# for j in range(1000, 1500, CONFIG.STEP):
				input_pnts.append((i,j))
		print("Total input length: {}".format(len(input_pnts)))

		test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
		predict_fn = theano.function([input_var], test_prediction)

		pos_pnt = []
		for index in range(0, len(input_pnts), self.BATCH_SIZE):
			print 'Batch {}'.format(index+1)
			inputs = [img.get_window(item) for item in input_pnts[index:index+self.BATCH_SIZE]]
			result = predict_fn(np.array(inputs).reshape(-1,1,CONFIG.AREA_SIZE,CONFIG.AREA_SIZE))
			# print(result)
			for tmp_index, tag in enumerate(result):
				if np.argmax(tag) == 1:
					pos_pnt.append((input_pnts[tmp_index+index], tag[1]))
		print 'Positive Point: {}'.format(len(pos_pnt))
		rects = []
		for pnt in pos_pnt:
			rects.append(plt.Circle(pnt[0], 20, facecolor=plt.cm.Greens(pnt[1]), alpha=0.3))
			# rects.append(
			# 	plt.Rectangle((pnt[0]-CONFIG.HALF_AREA_SIZE, pnt[1]-CONFIG.HALF_AREA_SIZE),
			# 		CONFIG.AREA_SIZE, CONFIG.AREA_SIZE,
			# 		facecolor='none', edgecolor='b', alpha=0.5))
		for pnt in img.tag:
			rects.append(
				plt.Rectangle((pnt[0]-CONFIG.HALF_AREA_SIZE, pnt[1]-CONFIG.HALF_AREA_SIZE),
					CONFIG.AREA_SIZE, CONFIG.AREA_SIZE,
					facecolor='none', edgecolor='r', alpha=1))

		fig = plt.figure()
		plt.imshow(img.image_data, cmap=plt.cm.gray)
		for rect in rects:
			fig.add_subplot(111).add_artist(rect)
		plt.savefig('window_predictions.png')
		plt.close(fig)

		positive_group = []
		for pnt in pos_pnt:
			x, y = pnt[0]
			flag = False
			for index, grp in enumerate(positive_group):
				c_x, c_y = grp[0]
				if (x-c_x)**2 + (y-c_y)**2 < (CONFIG.STEP*3)**2:
					weight = grp[1]
					new_cent = (float(x+c_x*weight)/float(weight+1), float(y+c_y*weight)/float(weight+1))
					positive_group[index] = (new_cent, weight+1, grp[2]+pnt[1]-0.5)
					flag = True
			if not flag:
				positive_group.append((pnt[0], 1, pnt[1]))

		cnt_positive = [0 for i in range(10)]
		cnt_positive_confid = [0 for i in range(50)]
		tag_flag = [[0 for i in range(len(img.tag))] for j in range(10)]
		tag_flag_confid = [[0 for i in range(len(img.tag))] for j in range(50)]

		pos_weight = []
		pos_confid = []
		neg_weight = []
		neg_confid = []
		for grp in positive_group:
			x, y = grp[0]
			flag = []
			for index, pnt in enumerate(img.tag):
				if (x-pnt[0])**2 + (y-pnt[1])**2 < (CONFIG.THRESHOLD)**2:
					flag.append(index)
			if flag:
				for wt in range(10):
					if grp[1] > wt:
						cnt_positive[wt] += 1
						for ind in flag:
							tag_flag[wt][ind] = 1
				for conf in range(50):
					if grp[2] > conf*0.05:
						cnt_positive_confid[conf] += 1
						for ind in flag:
							tag_flag_confid[conf][ind] = 1

				pos_weight.append(grp[1])
				pos_confid.append(grp[2])
				# cnt_positive += 1
			else:
				neg_weight.append(grp[1])
				neg_confid.append(grp[2])
		f_score = []
		for wt in range(10):
			print("Weight Threshold {}:".format(wt))
			precision = float(cnt_positive[wt])/float(len(filter(lambda x: x[1]>wt, positive_group)))
			recall = np.sum(tag_flag[wt])/float(len(img.tag))
			print("\tPrecision: {:.2f}".format(precision*100.))
			print("\tRecall: {:.2f}".format(recall*100.))
			f_score.append(2.*precision*recall/(precision+recall))
			print("\tF-score: {:.5f}".format(2.*precision*recall/(precision+recall)))

		fig = plt.figure()
		plt.plot(np.arange(0,10,1), f_score)
		plt.xlabel('Weight Threshold')
		plt.ylabel('F-Score')
		plt.savefig('f_score_weight.png')
		plt.close(fig)

		f_score = []
		for conf in range(50):
			print("Confidence Threshold {:.2f}:".format(conf*0.05))
			precision = float(cnt_positive_confid[conf])/float(len(filter(lambda x: x[2]>conf*0.05, positive_group)))
			recall = np.sum(tag_flag_confid[conf])/float(len(img.tag))
			print("\tPrecision: {:.2f}%".format(precision*100.))
			print("\tRecall: {:.2f}%".format(recall*100.))
			f_score.append(2.*precision*recall/(precision+recall))
			print("\tF-score: {:.5f}".format(2.*precision*recall/(precision+recall)))

		fig = plt.figure()
		plt.plot(np.arange(0,2.5,0.05), f_score)
		plt.xlabel('Confidence Threshold')
		plt.ylabel('F-Score')
		plt.savefig('f_score_confid.png')
		plt.close(fig)

		fig = plt.figure()
		rects = []
		for pnt in img.tag:
			rects.append(plt.Circle(pnt, 20, facecolor='g', alpha=0.5))
		for pnt in positive_group:
			rects.append(plt.Circle(pnt[0], 20, facecolor='b', alpha=0.5))
		fig = plt.figure()
		plt.imshow(img.image_data, cmap=plt.cm.gray)
		for rect in rects:
			fig.add_subplot(111).add_artist(rect)
		plt.savefig('group_prediction.png')
		plt.close(fig)


		fig = plt.figure()
		plt.hist(pos_weight, 10, facecolor='g', alpha=0.5)
		plt.hist(neg_weight, 10, facecolor='b', alpha=0.5)
		plt.savefig('weight_hist.png')
		plt.close(fig)

		fig = plt.figure()
		plt.hist(pos_confid, 50, facecolor='g', alpha=0.5)
		plt.hist(neg_confid, 50, facecolor='b', alpha=0.5)
		plt.savefig('confid_hist.png')
		plt.close(fig)

	def test_result(self, model_name):
		for img in self.test_set:
			img.extend_image()

		input_var = T.tensor4('inputs')
		target_var = T.ivector('targets')

		self.network = self.build_cnn(input_var)
		with np.load(model_name) as f:
		    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
		lasagne.layers.set_all_param_values(self.network, param_values)
		test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
		predict_fn = theano.function([input_var], test_prediction)

		precision_log = [[0. for i in range(len(self.test_set))] for j in range(50)]
		recall_log = [[0. for i in range(len(self.test_set))] for j in range(50)]
		f_score_log = [[0. for i in range(len(self.test_set))] for j in range(50)]
		for img_index, img in enumerate(self.test_set):
			print("Process image {} of {}".format(img_index+1, len(self.test_set)))
			input_pnts = []
			for i in range(CONFIG.HALF_AREA_SIZE, img.img_dim[0]-CONFIG.HALF_AREA_SIZE, CONFIG.STEP):
			# for i in range(1000, 1500, CONFIG.STEP):
				for j in range(CONFIG.HALF_AREA_SIZE, img.img_dim[1]-CONFIG.HALF_AREA_SIZE, CONFIG.STEP):
				#  for j in range(1000, 1500, CONFIG.STEP):
					input_pnts.append((i,j))
			print("Total input length: {}".format(len(input_pnts)))

			pos_pnt = []
			for index in range(0, len(input_pnts), self.BATCH_SIZE):
				print 'Batch {}/{}'.format(index,len(input_pnts))
				inputs = [img.get_window(item) for item in input_pnts[index:index+self.BATCH_SIZE]]
				result = predict_fn(np.array(inputs).reshape(-1,1,CONFIG.AREA_SIZE,CONFIG.AREA_SIZE))
				# print(result)
				for tmp_index, tag in enumerate(result):
					if np.argmax(tag) == 1:
						pos_pnt.append((input_pnts[tmp_index+index], tag[1]))
			print 'Positive Point: {}'.format(len(pos_pnt))

			positive_group = []
			for pnt in pos_pnt:
				x, y = pnt[0]
				flag = False
				for index, grp in enumerate(positive_group):
					c_x, c_y = grp[0]
					if (x-c_x)**2 + (y-c_y)**2 < (CONFIG.STEP*3)**2:
						weight = grp[1]
						new_cent = (float(x+c_x*weight)/float(weight+1), float(y+c_y*weight)/float(weight+1))
						positive_group[index] = (new_cent, weight+1, grp[2]+pnt[1]-0.5)
						flag = True
				if not flag:
					positive_group.append((pnt[0], 1, pnt[1]))

			cnt_positive_confid = [0 for i in range(50)]
			tag_flag_confid = [[0 for i in range(len(img.tag))] for j in range(50)]

			for grp in positive_group:
				x, y = grp[0]
				flag = []
				for index, pnt in enumerate(img.tag):
					if (x-pnt[0])**2 + (y-pnt[1])**2 < (CONFIG.THRESHOLD)**2:
						flag.append(index)
				if flag:
					for conf in range(50):
						if grp[2] > conf*0.05:
							cnt_positive_confid[conf] += 1
							for ind in flag:
								tag_flag_confid[conf][ind] = 1

			for conf in range(50):
				print("Confidence Threshold {:.2f}:".format(conf*0.05))
				total_num_grp = len(filter(lambda x: x[2]>conf*0.05, positive_group))
				if total_num_grp != 0:
					precision = float(cnt_positive_confid[conf])/float(total_num_grp)
				else:
					precision = 1e-10
				recall = np.sum(tag_flag_confid[conf])/float(len(img.tag))
				f_score = 2.*precision*recall/(precision+recall)
				print("\tPrecision: {:.2f}%".format(precision*100.))
				print("\tRecall: {:.2f}%".format(recall*100.))
				print("\tF-score: {:.5f}".format(f_score))
				precision_log[conf][img_index] = precision
				recall_log[conf][img_index] = recall
				f_score_log[conf][img_index] = f_score
		print(precision_log)
		print(recall_log)
		print(f_score_log)

		precision_result = []
		recall_result = []
		f_score_result = []
		for index in range(50):
			precision_result.append(np.average(precision_log[index]))
			recall_result.append(np.average(recall_log[index]))
			f_score_result.append(np.average(f_score_log[index]))
		print('Maximum F-score: {}'.format(np.max(f_score_result)))
		plt.figure()
		ind = np.arange(0,2.5,0.05)
		plt.plot(ind, precision_result, label='precision')
		plt.plot(ind, recall_result, label='recall')
		plt.plot(ind, f_score_result, label='f-score')
		plt.legend()
		plt.xlabel('Confidence Threshold')
		plt.savefig('result.png')

	def load_model(self, model_name):
		input_var = T.tensor4('inputs')
		target_var = T.ivector('targets')

		self.network = self.build_cnn(input_var)
		with np.load(model_name) as f:
		    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
		lasagne.layers.set_all_param_values(self.network, param_values)

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
				network, num_filters=48, filter_size=(11, 11), stride=(4,4),
				nonlinearity=lasagne.nonlinearities.rectify,
				W=lasagne.init.GlorotUniform())
	    # Expert note: Lasagne provides alternative convolutional layers that
    	# override Theano's choice of which implementation to use; for details
    	# please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

	    # Max-pooling layer of factor 2 in both dimensions:
		network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

	    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
		network = lasagne.layers.Conv2DLayer(
				network, num_filters=128, filter_size=(5, 5),
				nonlinearity=lasagne.nonlinearities.rectify)
		network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

	    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
		network = lasagne.layers.Conv2DLayer(
				network, num_filters=197, filter_size=(3, 3),
				nonlinearity=lasagne.nonlinearities.rectify)
		network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

	    #  A fully-connected layer of 256 units with 50% dropout on its inputs:
		network = lasagne.layers.DenseLayer(
				lasagne.layers.dropout(network, p=.5),
				num_units=512,
				nonlinearity=lasagne.nonlinearities.rectify)

    	# And, finally, the 10-unit output layer with 50% dropout on its inputs:
		network = lasagne.layers.DenseLayer(
				lasagne.layers.dropout(network, p=.5),
				num_units=2,
				nonlinearity=lasagne.nonlinearities.softmax)

		return network
