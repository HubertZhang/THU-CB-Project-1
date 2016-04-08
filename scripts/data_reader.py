#!/usr/bin/python

from numpy import *
import struct
import os
import matplotlib.pyplot as plt
import random
import scipy.spatial
import slist
import CONFIG

class Dataset(slist.SList):
    def __init__(self, data_root):
        self.data_root = data_root

    def setup_dataset(self):
        for item in os.listdir(self.data_root):
            file_info = item.split('.')
            if not len(file_info) == 2:
                print('Please check file: {}'.format(item))
                return
            data_name = file_info[0]
            file_type = file_info[1]
            if file_type == 'mrc':
                self.append(self.process_single_data(data_name))

    def process_single_data(self, data_name):
        data_item = DataItem(self.data_root, data_name)
        data_item.read_image()
        data_item.read_tag()
        return data_item


class DataItem():
    def __init__(self, data_root, data_name):
        self.NUMBYTES1 = 56
        self.NUMBYTES2 = 80 * 10
        self.data_name = data_name
        self.data_root = data_root
        self.tag = slist.SList([])

    def read_image(self):
        with open(os.path.join(self.data_root, '{}.mrc'.format(self.data_name)), 'rb') as input_image:
            self.img_header1 = input_image.read(self.NUMBYTES1 * 4)
            self.img_header2 = input_image.read(self.NUMBYTES2)

            byte_pattern = '=' + 'l' * self.NUMBYTES1  # '=' required to get machine independent standard size
            self.img_dim = struct.unpack(byte_pattern, self.img_header1)[:3]  # (dimx,dimy,dimz)
            self.img_type = struct.unpack(byte_pattern, self.img_header1)[3]  # 0: 8-bit signed, 1:16-bit signed, 2: 32-bit float, 6: unsigned 16-bit (non-std)
            if (self.img_type == 0):
                imtype = 'b'
            elif (self.img_type == 1):
                imtype = 'h'
            elif (self.img_type == 2):
                imtype = 'f4'
            elif (self.img_type == 6):
                imtype = 'H'
            else:
                type = 'unknown'  # should put a fail here
            input_image_dimension = (self.img_dim[1], self.img_dim[0])  # 2D images assumed

            self.image_data = fromfile(file=input_image, dtype=imtype, count=self.img_dim[0] * self.img_dim[1]).reshape(input_image_dimension)

    def read_tag(self):
        with open(os.path.join(self.data_root, '{}_manual_lgc.star'.format(self.data_name)), 'r') as input_tag:
            for line in input_tag.readlines():
                line = line[:-1]
                tmp_info = [x for x in line.split(' ') if x != '' ]
                if len(tmp_info) == 5:
                    self.tag.append([float(tmp_info[0]), float(tmp_info[1])])
        self.tag_tree = scipy.spatial.KDTree(self.tag)

    def generate_image(self):
        self.show_result(self.tag)

    def show_result(self, points, edge_x=10, edge_y=10):
        # circles = []
        rects = []
        for pnt in points:
            # circles.append(plt.Circle(pnt, 10, facecolor='none',alpha=1))
            rects.append(
                plt.Rectangle((pnt[0] - int(edge_x / 2), pnt[1] - int(edge_y / 2)), edge_x, edge_y, facecolor='none',
                              alpha=1))
        fig = plt.figure()
        plt.imshow(self.image_data, cmap=plt.cm.gray)
        # for circle in circles:
        #     fig.add_subplot(111).add_artist(circle)
        for rect in rects:
            fig.add_subplot(111).add_artist(rect)
        plt.show()

    def show_training_set(self, points, edge_x=10, edge_y=10):
        # circles = []
        rects = []
        for point in points:
            pnt = point[0]
            color = 'green' if point[1] else 'red'
            # circles.append(plt.Circle(pnt, 10, facecolor='none',alpha=1))
            rects.append(
                plt.Rectangle((pnt[0] - int(edge_x / 2), pnt[1] - int(edge_y / 2)), edge_x, edge_y, facecolor='none',
                              edgecolor=color, alpha=1))
        fig = plt.figure()
        plt.imshow(self.image_data, cmap=plt.cm.gray)
        # for circle in circles:
        #     fig.add_subplot(111).add_artist(circle)
        for rect in rects:
            fig.add_subplot(111).add_artist(rect)
        plt.show()

    def contain_tag(self, range_x, range_y):
        for tag in self.tag:
            if tag[0] in range_x and tag[1] in range_y:
                return True

    def is_positive(self, center_x, center_y):
        nearest = self.tag_tree.query((center_x, center_y))
        distance = ((nearest[0]-center_x) ** 2 + (nearest[1]-center_y) ** 2)
        return distance <= CONFIG.THRESHOLD*CONFIG.THRESHOLD

    def generate_training_set(self):

        step = CONFIG.STEP
        # changes = [(0, 0), (-step * 2, 0), (-step, 0), (-step, step),
        #           (0, step * 2), (0, step), (step, step),
        #           (step * 2, 0), (step, 0), (step, -step),
        #           (0, -step * 2), (0, -step), (-step, -step)]
        changes = [(0,0)]

        positive_set = []
        limit = CONFIG.STEP * 2 + CONFIG.HALF_AREA_SIZE
        for change in changes:
            for x in self.tag:
                if x[0]>=limit and x[0]<self.img_dim[0]-limit and x[1]>=limit and x[1]<self.img_dim[1]-limit:
                    positive_set.append(((x[0]+change[0],x[1]+change[1]), True))
        negative_set = []
        temp = sorted(self.tag)
        # while len(negative_set) != len(positive_set):
        for i in range(int(len(positive_set)*1.2)):
            x = random.randrange(CONFIG.HALF_AREA_SIZE, self.img_dim[0]-CONFIG.HALF_AREA_SIZE)
            y = random.randrange(CONFIG.HALF_AREA_SIZE, self.img_dim[1]-CONFIG.HALF_AREA_SIZE)
            if self.is_positive(x,y):
                continue
            negative_set.append(((x,y), False))
        total_set = positive_set + negative_set
        total_set = sorted(sorted(total_set, key=lambda x: x[0][1]), key=lambda x: x[0][0])
        return total_set

    def get_window(self, pnt):
        return self.image_data[pnt[0]-CONFIG.HALF_AREA_SIZE:pnt[0]+CONFIG.HALF_AREA_SIZE+1, pnt[1]-CONFIG.HALF_AREA_SIZE:pnt[1]+CONFIG.HALF_AREA_SIZE+1]
        # result = array([[0.0 for j in range(CONFIG.AREA_SIZE)] for i in range(CONFIG.AREA_SIZE)])
        # for i in range(CONFIG.AREA_SIZE):
        #     pnt_x = pnt[0]-CONFIG.HALF_AREA_SIZE+i
        #     for j in range(CONFIG.AREA_SIZE):
        #         pnt_y = pnt[1]-CONFIG.HALF_AREA_SIZE+j
        #         result[i][j] = self.image_data[pnt_x][pnt_y]
        # return result

    def generate_feature(self, dim_x, dim_y, step_x, step_y):
        def validate():
            return (pnt[0] < self.img_dim[0] - dim_x) and (pnt[1] < self.img_dim[1] - dim_y)

        def cut():
            tmp_feature = []
            for i in range(pnt[0], pnt[0] + dim_x):
                for j in range(pnt[1], pnt[1] + dim_y):
                    tmp_feature.append(self.image_data[i][j])

            self.feature_set.append(array(tmp_feature))
            self.label_set.append(self.contain_tag(range(pnt[0], pnt[0] + dim_x), range(pnt[1], pnt[1] + dim_y)))

        pnt = [0, 0]
        self.feature_set = []
        self.label_set = []
        while validate():
            while validate():
                cut()
                pnt[1] = pnt[1] + step_y
            pnt[0] = pnt[0] + step_x
            pnt[1] = 0
