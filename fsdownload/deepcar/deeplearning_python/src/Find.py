# -*- coding:utf-8 -*-
"""
get image from camera:/dev/video2  424*240

deal 128 *128     256*256

get the angle     object_detect
"""

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
import multiprocessing
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
import serial
ser = serial.Serial('/dev/ttyUSB0', 38400)


path = os.path.split(os.path.realpath(__file__))[0]+"/.."
opts,args = getopt.getopt(argv[1:],'-hH',['save_path=','vels=','camera='])

camera = "/dev/video2"
save_path = 'model_infer'
vels  = 1510
crop_size = 128

for opt_name,opt_value in opts:
    if opt_name in ('-h','-H'):
        print("python3 Auto_Driver.py --save_path=%s  --vels=%d --camera=%s "%(save_path , vels , camera))
        exit()
        
    if opt_name in ('--save_path'):
        save_path = opt_value

    if opt_name in ('--vels'):
       vels = int(opt_value)
       
    if opt_name in ('--camera'):
       camera = opt_value


def dataset(video):
    lower_hsv = np.array([20, 43, 46])
    upper_hsv = np.array([40, 255, 255])
    select.select((video,), (), ())   
    image_data = video.read_and_queue()
    frame = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)

    cv2.imwrite("003.jpg", frame)
    '''load  128*128'''
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask0 = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)   
    mask = mask0 #+ mask1
    img = Image.fromarray(mask)
    img = img.resize((128, 128), Image.ANTIALIAS)
    #img = cv2.resize(img, (128, 128))
    img = np.array(img).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
#    print("image____shape:",img.shape)
    '''object   256*256'''
    return img



'''##########################################################object  detect##########################################################'''

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
    "freeze_dir": "target2_model",
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


def load_model_detect():
    ues_tiny = train_parameters['use_tiny']
    yolo_config = train_parameters['yolo_tiny_cfg'] if ues_tiny else train_parameters['yolo_cfg']
    target_size = yolo_config['input_size']
    anchors = yolo_config['anchors']
    anchor_mask = yolo_config['anchor_mask']
    label_dict = train_parameters['num_dict']
    class_dim = train_parameters['class_dim']
    
    path1 = train_parameters['freeze_dir']
    model_dir = path1
    pm_config1 = pm.PaddleMobileConfig()
    pm_config1.precision = pm.PaddleMobileConfig.Precision.FP32######ok
    pm_config1.device = pm.PaddleMobileConfig.Device.kFPGA######ok
    #pm_config.prog_file = model_dir + '/model'
    #pm_config.param_file = model_dir + '/params'
    pm_config1.model_dir = model_dir
    pm_config1.thread_num = 4    
    predictor1 = pm.CreatePaddlePredictor(pm_config1)
    
    return predictor1
    
if __name__ == "__main__":
    cout_i = 0
    cout = 0
    flag = 0
    flag1 = 0
    flag2 = 0
    save_path  = path + "/model/" + save_path
    video = v4l2capture.Video_device(camera)
    video.set_format(320,240, fourcc='MJPG')
    video.create_buffers(1)
    video.queue_all_buffers()
    video.start()

    '''##########################################################object  detect##########################################################'''
    init_train_parameters()
    predictor1 = load_model_detect()    

    lib_path = path + "/lib" + "/libart_driver.so"
    so = cdll.LoadLibrary
    lib = so(lib_path)
    car = "/dev/ttyUSB0"
    z = np.zeros((1, 128, 128, 3))
    if (lib.art_racecar_init(38400, car.encode("utf-8")) < 0):
        raise
        pass
    #try:

    while 1:
        a = 1500
        vel = 1510
        while 1:
#            lib.send_cmd(1500, 1500)
            select.select((video,), (), ())
            image_data = video.read_and_queue()
            frame = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            cv2.imwrite("005.jpg", frame)
            origin = Image.open('005.jpg')

            tensor_img = origin.resize((256, 256), Image.BILINEAR)  #######resize 256*256

            if tensor_img.mode != 'RGB':
                tensor_img = tensor_img.convert('RGB')
            tensor_img = np.array(tensor_img).astype('float32').transpose((2, 0, 1))  # HWC to CHW
            tensor_img -= 127.5
            tensor_img *= 0.007843
            tensor_img = tensor_img[np.newaxis, :]

            tensor = pm.PaddleTensor()
            tensor.dtype = pm.PaddleDType.FLOAT32
            tensor.shape = (1, 3, 256, 256)
            tensor.data = pm.PaddleBuf(tensor_img)

            paddle_data_feeds1 = [tensor]
            outputs1 = predictor1.Run(paddle_data_feeds1)

            assert len(outputs1) == 1, 'error numbers of tensor returned from Predictor.Run function !!!'
            bboxes = np.array(outputs1[0], copy=False)

            t_labels = []
            t_scores = []
            t_boxes = []
            center_x = 0
            center_y = 0
            center_w = 0
            center_h = 0
            area = 0

            if len(bboxes.shape) == 1:
                print("No object found in video")
                lib.send_cmd(1500, 1500)
            else:
                labels = bboxes[:, 0].astype('int32')
                scores = bboxes[:, 1].astype('float32')
                boxes = bboxes[:, 2:].astype('float32')

                value_labels = []
                value_scoers = []
                value_boxes = []
                for i in range(len(labels)):
                    if scores[i] > 0.5:
                        value_labels.append(labels[i])
                        value_scoers.append(scores[i])
                        value_boxes.append(boxes[i])
                for i in range(len(value_labels)):
                    box = value_boxes[i]
                    for index in range(len(box)):
                        if index == 0 or index == 2:
                            value_boxes[i][index] = (value_boxes[i][index] / 608) * 320
                        elif index == 1 or index == 3:
                            value_boxes[i][index] = (value_boxes[i][index] / 608) * 240
#                draw_bbox_image(origin, value_boxes, value_labels, value_scoers)

                target_scoers = []
                target_boxes = []
                for i in range(len(value_labels)):
                    if int(value_labels[i]) == 0:
                        target_scoers.append(value_scoers[i])
                        target_boxes.append(value_boxes[i])

                if len(target_scoers) != 0:
                    list_s = list(target_scoers)
                    max_score = max(list_s)
                    index = list_s.index(max_score)
                    max_box = target_boxes[index]
                    center_x = int((max_box[0] + max_box[2]) / 2)
                    center_y = int((max_box[1] + max_box[3]) / 2)
                    center_w = int((max_box[0] - max_box[2]))
                    center_h = int((max_box[1] - max_box[3]))
                    center_w = abs(center_w)
                    center_h = abs(center_h)
                    area = center_w * center_h
#                    print("center_x, center_y", center_x, center_y)
#                    print("center_h, center_w:", center_h, center_w)
#                    print("area:", area)
                    if area > 4200:
                        vels = 1500
                    elif area > 2500:
                        vels = 1510
                    elif area > 300:
                        vels = 1510
                    else:
                        vels = 1510
                    angle = center_x - 160
                    angle = -angle
                    print(angle)
                    a = int(angle * 5 + 1500)
                    vel = int(vels)
                    lib.send_cmd(vel, a)

