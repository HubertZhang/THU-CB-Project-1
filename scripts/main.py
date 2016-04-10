import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

import data_reader
import worker
import CONFIG

logging.basicConfig(filename='app_info.log', level=logging.INFO)
logging.info("Parameters:")
logging.info("\tSTEP: {}".format(CONFIG.STEP))
logging.info("\tAREA_SIZE: {}".format(CONFIG.AREA_SIZE))
logging.info("\tTHRESHOLD: {}".format(CONFIG.THRESHOLD))
logging.info("\tLEARNING_RATE: {}".format(CONFIG.LEARNING_RATE))
logging.info("\tMOMENTUM: {}".format(CONFIG.MOMENTUM))

if len(sys.argv) != 3:
	print('Usage: python main.py <data_path> <output_path>')
	sys.exit(0)

data_root = os.path.abspath(sys.argv[1])
output_path = os.path.abspath(sys.argv[2])

training_data = data_reader.Dataset(os.path.join(data_root, 'train'))
training_data.setup_dataset()
testing_data = data_reader.Dataset(os.path.join(data_root, 'test'))
testing_data.setup_dataset()

if not os.path.isdir(output_path):
	print('Output path does not exist, create a new one.')
	os.mkdir(output_path)

os.chdir(output_path)
# alg = analyzer.Analyzer(training_data, testing_data)
# alg.main()
alg = worker.Worker(training_data, testing_data)
# alg.main()
# alg.load_model('model_2.npz')
# alg.predict(alg.training_set[-1],'model.npz')
alg.test_result('model.npz')
# alg.main_minst()


def print_image(img_mat, fig_name):
	fig = plt.figure()
	plt.imshow(img_mat, cmap=plt.cm.gray)
	plt.savefig(fig_name)
	plt.close(fig)

def generate_feature(data_item):
	def cut_mat(dim_x, dim_y):
		tmp_feature = []
		for x in range(max(0, int(pnt[0]-dim_x)), min(data_item.img_dim[0], int(pnt[0]+dim_x))+1):
			for y in range(max(0, int(pnt[1]-dim_y)), min(data_item.img_dim[1], int(pnt[1]+dim_y))+1):
				tmp_feature.append(image_data[x][y])
		return tmp_feature

	image_data = data_item.image_data
	feature_set = []
	tag_set = []
	for index, pnt in enumerate(data_item.tag):
		new_mat = cut_mat(100, 100)

# generate_feature(training_data[0])