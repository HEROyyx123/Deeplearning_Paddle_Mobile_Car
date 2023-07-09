import os
import v4l2capture
from ctypes import *
import struct, array
from fcntl import ioctl
import cv2
import numpy as np
import time
from sys import argv
import getopt
import sys, select, termios, tty
import threading
import paddlemobile as pm
from paddlelite import *
import codecs
#import paddle
import multiprocessing
#import paddle.fluid as fluid
#from IPython.display import display
import math
import functools
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import ImageFont
from PIL import ImageDraw
from collections import namedtuple
from datetime import datetime 
from user import user_cmd
import socket

path = os.path.split(os.path.realpath(__file__))[0]+"/.."
obj_path = 'freeze_model'
vels = 1535
opts, args = getopt.getopt(argv[1:], '-hH', ['save_path=', 'vels=', 'camera='])

train_parameters ={
    "train_list": "train.txt",
    "eval_list": "eval.txt",
    "class_dim": -1,
    "label_dict": {},
    "num_dict": {},
    "image_count": -1,
    "continue_train": True,     # 是否加载前一次的训练参数，接着训练
    "pretrained": False,
    "pretrained_model_dir": "./pretrained-model",
    "save_model_dir": "./yolo-model",
    "model_prefix": "yolo-v3",
    "freeze_dir": "freeze_model6",
    #"freeze_dir": "../model/tiny-yolov3",
    "use_tiny": True,          # 是否使用 裁剪 tiny 模型
    "max_box_num": 2,          # 一幅图上最多有多少个目标
    "num_epochs": 80,
    "train_batch_size": 32,      # 对于完整 yolov3，每一批的训练样本不能太多，内存会炸掉；如果使用 tiny，可以适当大一些
    "use_gpu": False,
    "yolo_cfg": {
        "input_size": [3, 448, 448],    # 原版的边长大小为608，为了提高训练速度和预测速度，此处压缩为448
        "anchors": [7, 10, 12, 22, 24, 17, 22, 45, 46, 33, 43, 88, 85, 66, 115, 146, 275, 240],
        "anchor_mask": [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    },
    "yolo_tiny_cfg": {
        "input_size": [3, 224, 224],
        "anchors": [6, 8, 13, 15, 22, 34, 48, 50, 81, 100, 205, 191],
        "anchor_mask": [[3, 4, 5], [0, 1, 2]]
    },
    "ignore_thresh": 0.7,
    "mean_rgb": [127.5, 127.5, 127.5],
    "mode": "train",
    "multi_data_reader_count": 4,
    "apply_distort": True,
    "nms_top_k": 300,
    "nms_pos_k": 300,
    "valid_thresh": 0.01,
    "nms_thresh": 0.45,
    "image_distort_strategy": {
        "expand_prob": 0.5,
        "expand_max_ratio": 4,
        "hue_prob": 0.5,
        "hue_delta": 18,
        "contrast_prob": 0.5,
        "contrast_delta": 0.5,
        "saturation_prob": 0.5,
        "saturation_delta": 0.5,
        "brightness_prob": 0.5,
        "brightness_delta": 0.125
    },
    "sgd_strategy": {
        "learning_rate": 0.002,
        "lr_epochs": [30, 50, 65],
        "lr_decay": [1, 0.5, 0.25, 0.1]
    },
    "early_stop": {
        "sample_frequency": 50,
        "successive_limit": 3,
        "min_loss": 2.5,
        "min_curr_map": 0.84
    }
}
def init_train_parameters():
    """
    初始化训练参数，主要是初始化图片数量，类别数
    :return:
    """
    file_list = "./data/data6045/train.txt"#os.path.join(train_parameters['data_dir'], train_parameters['train_list'])
    label_list =  "./data/data6045/label_list"#os.path.join(train_parameters['data_dir'], "label_list")
    index = 0
    with codecs.open(label_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        for line in lines:
            train_parameters['num_dict'][index] = line.strip()
            train_parameters['label_dict'][line.strip()] = index
            index += 1
        train_parameters['class_dim'] = index
    with codecs.open(file_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        train_parameters['image_count'] = len(lines)



def load_model_detect(obj_path):
    # 加载目标检测模型和模型参数

    path1 = train_parameters['freeze_dir']
    model_dir = path1
    print(model_dir)
    pm_config1 = pm.PaddleMobileConfig()
    pm_config1.precision = pm.PaddleMobileConfig.Precision.FP32
    pm_config1.device = pm.PaddleMobileConfig.Device.kFPGA

    pm_config1.model_dir = model_dir
    pm_config1.thread_num = 4
    predictor1 = pm.CreatePaddlePredictor(pm_config1)

    return predictor1


def objective():
    # 目标检测进程

    global vels
    global path, obj_path
    obj_path = path + "/model/" + obj_path

    socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    socket_server.bind(('', 1234))
    socket_server.listen(28)
    client_socket, client_address = socket_server.accept()
    print('Connection successful')

    predictor1 = load_model_detect(obj_path)
    while 1:
        try:
            origin = Image.open(path + "/1.jpg")

            tensor_img = origin.resize((224, 224), Image.BILINEAR)  #######resize 256*256

            if tensor_img.mode != 'RGB':
                tensor_img = tensor_img.convert('RGB')
            tensor_img = np.array(tensor_img).astype('float32').transpose((2, 0, 1))  # HWC to CHW
            tensor_img -= 127.5
            tensor_img *= 0.007843
            tensor_img = tensor_img[np.newaxis, :]

            tensor = pm.PaddleTensor()
            tensor.dtype = pm.PaddleDType.FLOAT32
            tensor.shape = (1, 3, 224, 224)
            tensor.data = pm.PaddleBuf(tensor_img)
            paddle_data_feeds1 = [tensor]
            outputs1 = predictor1.Run(paddle_data_feeds1)

            assert len(outputs1) == 1, 'error numbers of tensor returned from Predictor.Run function !!!'
            bboxes = np.array(outputs1[0], copy=False)
            print("bboxes.shape", bboxes.shape)

            if len(bboxes.shape) == 1:
                send_data = 'null'
                client_socket.send(send_data.encode('utf-8'))
                print("No object found in video")
            else:
                labels = bboxes[:, 0].astype('int32')
                scores = bboxes[:, 1].astype('float32')
                boxes = bboxes[:, 2:].astype('float32')

                list_s = list(scores)
                max_score = max(list_s)
                print(labels, max_score)
                index = list_s.index(max_score)

                if max_score > 0.70:
                    max_box = boxes[index]
                    max_label = int(labels[index])
                    print(max_label)
                    center_x= int((max_box[0] + max_box[2]) / 2)
                    center_y = int((max_box[1] + max_box[3]) / 2)
                    print(center_x, center_y)
                    if max_label == 0:
                        if(center_y >= 300):
                            vels = 1530
                    elif max_label == 1:
                        if(center_y >= 430):
                            vels = 1560
                    elif max_label == 2:
                        if(center_y >= 250):
                            vels = 1490
                    elif max_label == 3:
                        if(center_y >= 220):
                            vels = 1560
                    elif max_label == 4:
                        vels = 1560
                    elif max_label == 5:
                        if(center_x>=320 and center_y>=230 and center_x<=450):
                            vels = 1487
                    else:
                        vels = 1560
                    send_data = str(vels)
                    client_socket.send(send_data.encode('utf-8'))
                    print(send_data)
        except:
            print('error')

if __name__ == '__main__':

    while 1:
        init_train_parameters()
        objective()