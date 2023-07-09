# -*- coding:utf-8 -*-

"""
Three process runs
A process captures images
A process run lane line model, get the Angle
A process runs the Yolov3 target detection model to get the target category and location
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

path = os.path.split(os.path.realpath(__file__))[0]+"/.."
opts,args = getopt.getopt(argv[1:],'-hH',['save_path=', 'obj_path=', 'vels=','camera='])


save_path = 'model_5'
obj_path = 'freeze_model6'
vels = 1535
crop_size = 128

camera = multiprocessing.Array("b",range(50))
camera.value = "/dev/video2"

lock = multiprocessing.Manager().Lock()
lock2 = multiprocessing.Manager().Lock()

for opt_name,opt_value in opts:
    # 终端获取参数

    if opt_name in ('-h','-H'):
        print("python3 Auto_Driver.py --save_path=%s --obj_path=%s --vels=%d --camera=%s "%(save_path ,obj_path ,vels , camera))
        exit()
        
    if opt_name in ('--save_path'):
        save_path = opt_value

    if opt_name in ('--obj_path'):
        obj_path = opt_value

    if opt_name in ('--vels'):
       vels = int(opt_value)
       
    if opt_name in ('--camera'):
       camera = opt_value


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
    "freeze_dir": obj_path,
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
        "input_size": [3, 256, 256],
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


def save_image_process(lock, camera, state, state2):
    # 采集图片进程，并保存

    global path
    video = v4l2capture.Video_device(camera.value)
    video.set_format(320, 240, fourcc='MJPG')
    video.create_buffers(50)
    video.queue_all_buffers()
    video.start()
    while 1:
        while state.value == True and state2.value == True:
            select.select((video,), (), ())
            image_data = video.read_and_queue()
            frame1 = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            lock.acquire()
            cv2.imwrite(path + "/1.jpg", frame1)
            state.value = False
            state2.value = False
            lock.release()
            print('SAVE IMAGE TO LOCAL')


def dataset(frame):
    # 车道上图片预处理

    lower_hsv = np.array([26, 43, 46])
    upper_hsv = np.array([34, 255, 255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask0 = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
    mask = mask0

    img = Image.fromarray(mask)
    img = img.resize((128, 128), Image.ANTIALIAS)
    img = np.array(img).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img


def load_model():
    # 加载车道线模型和模型参数

    valid_places =   (
		Place(TargetType.kFPGA, PrecisionType.kFP16, DataLayoutType.kNHWC),
		Place(TargetType.kHost, PrecisionType.kFloat),
		Place(TargetType.kARM, PrecisionType.kFloat),
	)
    config = CxxConfig()
    model = save_path
    model_dir = model
    print(model_dir)
    config.set_model_file(model_dir + "/model")
    config.set_param_file(model_dir + "/params")
    config.set_valid_places(valid_places)
    predictor = CreatePaddlePredictor(config)
    return predictor


def predict(predictor, image, z):
    img = image

    i = predictor.get_input(0)
    i.resize((1, 3, 128, 128))
    z[ 0,0:img.shape[1], 0:img.shape[2] + 0, 0:img.shape[3]] = img
    z = z.reshape(1, 3, 128, 128)
    frame1 = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite("test.jpg", frame1)
    i.set_data(z)

    predictor.run()
    out = predictor.get_output(0)
    score = out.data()[0][0]
    print(out.data()[0])
    return score


def control_car_process(lock, vels, state, lock2, state3):
    # 控制车进程

    global path, save_path
    save_path = path + "/model/" + save_path
    cout = 0
    lock.acquire()
    predictor = load_model()
    lock.release()
    vel = int(vels.value)
    lib_path = path + "/lib" + "/libart_driver.so"
    so = cdll.LoadLibrary
    lib = so(lib_path)
    car = "/dev/ttyUSB0"
    if (lib.art_racecar_init(38400, car.encode("utf-8")) < 0):
        raise
        pass
    z = np.zeros((1, 128, 128, 3))
    while 1:
        while state.value == False:
            while state3.value == False:
                lock.acquire()
                frame = cv2.imread(path+"/1.jpg")
                state.value =True
                lock.release()
                img = dataset(frame)
                z = np.zeros((1, 128, 128, 3))
                lock2.acquire()
                state3.value = True
                angle = predict(predictor, img, z)
                a = int(angle * 3000 + 0)
                lib.send_cmd(vel, a)
                lock2.release()
                print(cout)
                cout = cout + 1

def init_train_parameters():
    # 初始化训练参数，主要是初始化图片数量，类别数

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

    path = train_parameters['freeze_dir']
    model_dir = path
    print(model_dir)
    pm_config1 = pm.PaddleMobileConfig()
    pm_config1.precision = pm.PaddleMobileConfig.Precision.FP32
    pm_config1.device = pm.PaddleMobileConfig.Device.kFPGA

    pm_config1.model_dir = model_dir
    pm_config1.thread_num = 4   
    predictor1 = pm.CreatePaddlePredictor(pm_config1)

    return predictor1


def obj_process(lock, state2, lock2, state3, vels):
    # 目标检测进程

    global path, obj_path
    init_train_parameters()
    obj_path = path + "/model/" + obj_path
    lock.acquire()
    predictor1 = load_model_detect(obj_path)
    lock.release()
    while 1:
        while state2.value == False:
            while state3.value == True:
                lock.acquire()
                origin = Image.open(path+"/1.jpg")
                state2.value = True

                lock.release()
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
                lock2.acquire()
                outputs1 = predictor1.Run(paddle_data_feeds1)
                state3.value = False
                lock2.release()
                assert len(outputs1) == 1, 'error numbers of tensor returned from Predictor.Run function !!!'
                bboxes = np.array(outputs1[0], copy=False)
                # print("bboxes.shape", bboxes.shape)

                if len(bboxes.shape) == 1:
                    print("No object found in video")
                else:
                    labels = bboxes[:, 0].astype('int32')
                    scores = bboxes[:, 1].astype('float32')
                    boxes = bboxes[:, 2:].astype('float32')

                    list_s = list(scores)
                    max_score = max(list_s)
                    # print(labels, max_score)
                    index = list_s.index(max_score)

                    if max_score > 0.65:
                        max_box = boxes[index]
                        max_label = int(labels[index])
                        print(max_label)
                        center_x = int((max_box[0] + max_box[2]) / 2)
                        center_y = int((max_box[1] + max_box[3]) / 2)
                        print(center_x, center_y)
                        if max_label == 0:
                            if (center_y >= 300):
                                vels.value = 1530
                        elif max_label == 1:
                            if (center_y >= 430):
                                vels.value = 1550
                        elif max_label == 2:
                            if (center_y >= 220):
                                vels.value = 1500
                        elif max_label == 3:
                            if (center_y >= 220):
                                vels.value = 1550
                        elif max_label == 4:
                            vels.value = 1550
                        elif max_label == 5:
                            if (center_x >= 320 and center_y >= 250):
                                vels.value = 1500
                        else:
                            vels.value = 1550


if __name__ == "__main__":

    STATE = multiprocessing.Value("i", True)
    STATE2 = multiprocessing.Value("i", True)
    STATE3 = multiprocessing.Value("i", False)
    vels = multiprocessing.Value("i", vels)

    try:

        process_image = multiprocessing.Process(target=save_image_process, args=(lock, camera, STATE, STATE2), name='process_image')
        process_car = multiprocessing.Process(target=control_car_process, args=(lock, vels, STATE, lock2, STATE3), name='process_car')
        process_obj = multiprocessing.Process(target=obj_process, args=(lock, STATE2, lock2, STATE3, vels), name='process_obj')
        process_image.start()
        process_car.start()
        process_obj.start()
        while 1:
            {}



    except:
        print('error')
    finally:
        lib.send_cmd(1500, 1500)
