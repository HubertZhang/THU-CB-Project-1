#!/usr/bin/python

from numpy import *
import struct
import os

import slist

class Dataset(slist.SList):
    def __init__(self, data_root):
        self.data_root = data_root

    def setup_dataset(self):
        for item in os.listdir(self.data_root):
            file_info = item.split('.')
            if not len(file_info) == 2:
                print 'Please check file: {}'.format(item)
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
        self.NUMBYTES2 = 80*10
        self.data_name = data_name
        self.data_root = data_root
        self.tag = slist.SList([])

    def read_image(self):
        with open(os.path.join(self.data_root,'{}.mrc'.format(self.data_name)),'rb') as input_image:
            self.img_header1 = input_image.read(self.NUMBYTES1*4)
            self.img_header2 = input_image.read(self.NUMBYTES2)

            byte_pattern = '=' + 'l' * self.NUMBYTES1   #'=' required to get machine independent standard size
            self.img_dim = struct.unpack(byte_pattern,self.img_header1)[:3]   #(dimx,dimy,dimz)
            self.img_type = struct.unpack(byte_pattern,self.img_header1)[3]  #0: 8-bit signed, 1:16-bit signed, 2: 32-bit float, 6: unsigned 16-bit (non-std)
            if (self.img_type == 0):
                imtype = 'b'
            elif (self.img_type ==1):
                imtype = 'h'
            elif (self.img_type ==2):
                imtype = 'f4'
            elif (self.img_type ==6):
                imtype = 'H'
            else:
                type = 'unknown'   #should put a fail here
            input_image_dimension = (self.img_dim[1],self.img_dim[0])  #2D images assumed

            self.image_data = fromfile(file=input_image,dtype=imtype,count=self.img_dim[0]*self.img_dim[1]).reshape(input_image_dimension)

    def read_tag(self):
        with open(os.path.join(self.data_root,'{}_manual_lgc.star'.format(self.data_name)),'r') as input_tag:
            for line in input_tag.read().split('\n'):
                tmp_info = filter(lambda x: x, line.split(' '))
                if len(tmp_info) == 5:
                    self.tag.append((tmp_info[0], tmp_info[1]))
