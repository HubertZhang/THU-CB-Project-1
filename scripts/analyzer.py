import random
import matplotlib.pyplot as plt
import numpy
from sklearn import svm
from sklearn import tree
from slist import SList
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage
from scipy import signal, misc
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SigmoidLayer
from sklearn.decomposition import PCA

class Analyzer:
	
	def __init__(self, training_set, test_set):
		self.training_set = training_set
		self.test_set = test_set
		self.MAT_SIZE = (50,50)
		self.POS_NEG_RATIO = 2
	
	def generate_feature(self, data_item):
		def cut_mat(dim_x, dim_y):
			tmp_feature = []
			for x in range(max(0, int(pnt[0]-dim_x)), min(data_item.img_dim[0]-1, int(pnt[0]+dim_x))+1):
				for y in range(max(0, int(pnt[1]-dim_y)), min(data_item.img_dim[1]-1, int(pnt[1]+dim_y))+1):
					tmp_feature.append(image_data[x][y])
			return numpy.array(tmp_feature)

		data_set = SList([])

		image_data = data_item.image_data
		for index, pnt in enumerate(data_item.tag):
			data_set.append((cut_mat(self.MAT_SIZE[0], self.MAT_SIZE[1]),1, pnt))

		for i in range(int(len(data_item.tag)*self.POS_NEG_RATIO)):
			pnt = (random.randint(self.MAT_SIZE[0], data_item.img_dim[0]-self.MAT_SIZE[0]), random.randint(self.MAT_SIZE[1], data_item.img_dim[1]-self.MAT_SIZE[1]))
			if data_item.contain_tag(range(pnt[0]-self.MAT_SIZE[0],pnt[0]+self.MAT_SIZE[0]), range(pnt[1]-self.MAT_SIZE[1],pnt[1]+self.MAT_SIZE[1])):
				data_set.append((cut_mat(self.MAT_SIZE[0], self.MAT_SIZE[1]), 1, pnt))
			else:
				data_set.append((cut_mat(self.MAT_SIZE[0], self.MAT_SIZE[1]), 0, pnt))

		print('{}/{}'.format(data_set.filter_by(lambda x: x[1] == 1).count(), data_set.count()))
		return data_set.filter_by(lambda x: len(x[0]) == (2*self.MAT_SIZE[0]+1)*(2*self.MAT_SIZE[1]+1))

	def clustering(self):
		test_img = self.training_set[0].image_data
		num_set = []
		for i in range(len(test_img)):
			for j in range(len(test_img[i])):
				num_set.append(test_img[i][j])
				test_img[i][j] = max(0, test_img[i][j]-47)
		print_image(test_img, 'cleared_image.png')
		plt.figure()
		n, bins, patches = plt.hist(num_set, 50, normed=1, facecolor='green', alpha=0.75)
		plt.savefig('hist.png')

	def svm(self):
		img = self.training_set[0]
		img.image_data = gaussian_filter(img.image_data, 15)
		# img_average = numpy.average(img.image_data)
		training_set = self.generate_feature(img)
		img = self.training_set[1]
		img.image_data = gaussian_filter(img.image_data, 15)
		test_set = self.generate_feature(img)

		pca = PCA(n_components = 20)
		pca.fit([item[0] for item in training_set]+[item[0] for item in test_set])
		pca_training = pca.transform([item[0] for item in training_set])
		# for img in training_set:
		# 	print_image(img[0].reshape(2*self.MAT_SIZE[0]+1,2*self.MAT_SIZE[1]+1), '{}_fig_{}_{}.png'.format(img[1], img[2][0], img[2][1]))
		# training_set = training_set.map(lambda x: (x[0]-img_average, x[1]))
		model = svm.SVC()
		# model = tree.DecisionTreeClassifier()
		model.fit(pca_training,numpy.array([item[1] for item in training_set]))

		training_result = model.predict(pca_training)
		hit = 0
		for index, tag in enumerate(training_result):
			if tag == training_set[index][1]:
				hit += 1
		print(float(hit) / float(len(training_set)))

		pca_test = pca.transform([item[0] for item in test_set])
		# test_set = test_set.map(lambda x: (x[0]-img_average, x[1]))
		predicted = model.predict(pca_test)

		hit = 0
		for index, tag in enumerate(predicted):
			if tag == test_set[index][1]:
				hit += 1
		print(float(hit) / float(len(test_set)))

	def gradient(self):
		image_data = self.training_set[0].image_data
		new_feature = []
		for row in range(len(image_data)):
			tmp_row = [[] for i in range(len(image_data[row]))]
			for col in range(len(tmp_row)):
				grad = [0, 0, 0]
				if row != 0:
					grad[0] = image_data[row][col]-image_data[row-1][col]
				# if row != self.training_set[0].img_dim[0] - 1:
				# 	grad[1] = image_data[row][col]-image_data[row+1][col]
				if col != 0:
					grad[1] = image_data[row][col]-image_data[row][col-1]
				# if col != self.training_set[0].img_dim[1] - 1:
				# 	grad[3] = image_data[row][col]-image_data[row][col+1]
				tmp_row[col] = numpy.array(grad)
			new_feature.append(numpy.array(tmp_row))
		new_feature = numpy.array(new_feature)

		plt.figure()
		plt.imshow(new_feature)
		plt.show()

	def image_feature(self):
		image = self.training_set[0]
		image.image_data = gaussian_filter(image.image_data, 20)
		image_data = image.image_data
		gradients = []
		for row in range(len(image_data)):
			for col in range(len(image_data[row])):
				if row != 0:
					gradients.append(image_data[row][col]-image_data[row-1][col])
				# if row != self.training_set[0].img_dim[0] - 1:
				# 	grad[1] = image_data[row][col]-image_data[row+1][col]
				if col != 0:
					gradients.append(image_data[row][col]-image_data[row][col-1])
				# if col != self.training_set[0].img_dim[1] - 1:
				# 	grad[3] = image_data[row][col]-image_data[row][col+1]
		# print_image(image.image_data, 'filtered_image.png')
		# print_image([item[0:2000] for item in image.image_data[0:2000]], 'test.png')
		# features = self.generate_feature(image)
		# for index, feature in enumerate(features):
		# 	print_image(feature[0].reshape(2*self.MAT_SIZE[0]+1, 2*self.MAT_SIZE[1]+1), '{}_fig_{}'.format(feature[1], index))
		gradients = filter(lambda x: abs(x)>0.02, gradients)
		plt.figure()
		n, bins, patches = plt.hist(gradients, 50, normed=1, facecolor='green', alpha=0.75)
		plt.savefig('gradient_hist.png')


	def neural_network(self):
		img = self.training_set[0]
		img.image_data = gaussian_filter(img.image_data, 15)
		# img_average = numpy.average(img.image_data)
		training_set = self.generate_feature(img)
		# for img in training_set:
		# 	print_image(img[0].reshape(2*self.MAT_SIZE[0]+1,2*self.MAT_SIZE[1]+1), '{}_fig_{}_{}.png'.format(img[1], img[2][0], img[2][1]))
		pca = PCA(n_components = 20)
		pca.fit([item[0] for item in training_set])
		pca_training = pca.transform([item[0] for item in training_set])

		feature_length = 20
		ds = SupervisedDataSet(feature_length, 1)
		for index, item in enumerate(training_set):
			ds.addSample(pca_training[index], (item[1],))
		# training_set = training_set.map(lambda x: (x[0]-img_average, x[1]))
		model = buildNetwork(feature_length, 2*feature_length, 1, bias=True, hiddenclass=SigmoidLayer)
		trainer = BackpropTrainer(model, ds)
		# model = tree.DecisionTreeClassifier()
		# model.fit(numpy.array([item[0] for item in training_set]),numpy.array([item[1] for item in training_set]))
		trainer.trainUntilConvergence()

		img = self.training_set[1]
		img.image_data = gaussian_filter(img.image_data, 15)
		test_set = self.generate_feature(img)
		pca_test = pca.transform([item[0] for item in test_set])
		# test_set = test_set.map(lambda x: (x[0]-img_average, x[1]))
		# predicted = model.activate([item[0] for item in test_set])
		hit = 0
		for index, item in enumerate(training_set):
			predicted_tag = model.activate(pca_training[index])
			if (predicted_tag[0]>0.5 and item[1]==1) or (predicted_tag[0]<=0.5 and item[1]==0):
				hit += 1
		print(float(hit) / float(len(training_set)))

		hit = 0
		for index, item in enumerate(test_set):
			predicted_tag = model.activate(pca_test[index])
			if (predicted_tag[0]>0.5 and item[1]==1) or (predicted_tag[0]<=0.5 and item[1]==0):
				hit += 1
		print(float(hit) / float(len(test_set)))

	def filter(self):
		# img_average = numpy.average(self.training_set[0].image_data)
		image = self.training_set[0].image_data
		derfilt = numpy.array([1.0,-2,1.0],numpy.float32)
		ck = signal.cspline2d(image,8.0)
		deriv = signal.sepfir2d(ck, derfilt, [1]) + signal.sepfir2d(ck, [1], derfilt)
		print_image(deriv, 'bsplines.png')
		for std in [1,2,4,8,16,32,64,128]:
			filtered_img = ndimage.median_filter(image, std)
			print_image(filtered_img, 'median_{}.png'.format(std))

	def new_svm(self):
		feature_set = []
		label_set = []
		for item in self.training_set:
			item.generate_feature(100, 100, 50, 50)
			feature_set += item.feature_set
			label_set += item.label_set
		split = int(len(feature_set)*0.8)

		training_feature = numpy.array(feature_set[:split])
		training_label = numpy.array(label_set[:split])
		print('Size of Training Set: {}'.format(len(training_feature)))
		validate_feature = numpy.array(feature_set[split:])
		validate_label = numpy.array(label_set[split:])
		print('Size of Validation Set: {}'.format(len(validate_feature)))

		model = svm.SVC()
		model.fit(training_feature, training_label)
		training_result = model.predict(training_feature)
		hit = 0
		for index in range(len(training_result)):
			if training_result[index] == training_label[index]:
				hit += 1
		print('Training Error: {}'.format(float(hit) / float(len(training_result))))

		validate_result = model.predict(validate_feature)
		hit = 0
		for index in range(len(validate_result)):
			if validate_result[index] == validate_label[index]:
				hit += 1
		print('Validation Error: {}'.format(float(hit) / float(len(validate_result))))

	def main(self):
		self.new_svm()
		# self.svm()
		# image = self.training_set[0]
		# image.image_data = gaussian_filter(image.image_data, 10)
		# image.show_result(image.tag)


def print_image(image_data, fig_name):
	plt.figure()
	plt.imshow(image_data, cmap=plt.cm.gray)
	plt.savefig(fig_name)
	plt.close()
