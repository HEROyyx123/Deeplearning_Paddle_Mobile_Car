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

import serial

ser = serial.Serial('/dev/ttyUSB0', 38400)


#script,vels,save_path= argv

path = os.path.split(os.path.realpath(__file__))[0]+"/.."
opts,args = getopt.getopt(argv[1:],'-hH',['save_path=','vels=','camera='])

camera = "/dev/video2"
save_path = 'model_18'
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
#def load_image(cap):

#    lower_hsv = np.array([156, 43, 46])
#    upper_hsv = np.array([180, 255, 255])
#    lower_hsv1 = np.array([0, 43, 46])
#    upper_hsv1 = np.array([10, 255, 255])
#    ref, frame = cap.read()


#    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   
#    mask0 = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
#    mask1 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=upper_hsv1)
#    mask = mask0 + mask1
#    img = Image.fromarray(mask)
#    img = img.resize((128, 128), Image.ANTIALIAS)
#    img = np.array(img).astype(np.float32)
#    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#    img = img.transpose((2, 0, 1))
#    img = img[(2, 1, 0), :, :] / 255.0
#    img = np.expand_dims(img, axis=0)
#    return img
def dataset(video):
    lower_hsv = np.array([20, 43, 46])
    upper_hsv = np.array([40, 255, 255])
    select.select((video,), (), ())   
    image_data = video.read_and_queue()
    frame = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)

#    frame = cv2.convertScaleAbs(frame, beta = -20)
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
    img = img / 255.0;
    img = np.expand_dims(img, axis=0)
#    print("image____shape:",img.shape)
    '''object   256*256'''
    return img;

def load_model():
    valid_places =   (
		Place(TargetType.kFPGA, PrecisionType.kFP16, DataLayoutType.kNHWC),
		Place(TargetType.kHost, PrecisionType.kFloat),
		Place(TargetType.kARM, PrecisionType.kFloat),
	);
    config = CxxConfig();
    model = save_path;
    model_dir = model;
    config.set_model_file(model_dir + "/model");
    config.set_param_file(model_dir + "/params");
    #config.model_dir = model_dir
    config.set_valid_places(valid_places);
    predictor = CreatePaddlePredictor(config);
    return predictor;
    
def predict(predictor, image, z):
    img = image; 

    i = predictor.get_input(0);
    i.resize((1, 3, 128, 128));
    z[ 0,0:img.shape[1], 0:img.shape[2] + 0, 0:img.shape[3]] = img
    z = z.reshape(1, 3, 128, 128);
    frame1 = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite("zzzzz_test.jpg", frame1)
    i.set_data(z)

    predictor.run();
    out = predictor.get_output(0);
    score = out.data()[0][0];
#    print(out.data()[0])
    return score;
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
    "freeze_dir": "freeze_model.8.18",
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


def read_image():
    img_path = "/home/root/workspace/deepcar/deeplearning_python/src/mmmmm2.jpg"
    
    lock.acquire()
    origin = Image.open(img_path)
    lock.release()  
  
    #origin = image
    #img = resize_img(origin, target_size)
    img = origin.resize((224,224), Image.BILINEAR)   #######resize 256*256
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img).astype('float32').transpose((2, 0, 1))  # HWC to CHW
    img -= 127.5
    img *= 0.007843
    img = img[np.newaxis, :]
    return img
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
    stop = 0

    save_path  = path + "/model/" + save_path
    video = v4l2capture.Video_device(camera)
    video.set_format(320,240, fourcc='MJPG')
    video.create_buffers(1)
    video.queue_all_buffers()
    video.start()
   
    predictor = load_model();
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
        last_label = 0
        i = 0
        count =0
        nowtime2 = datetime.now()
        while 1:
            FLAG = 0
            FLAg = 0
            angle1 = 0
            nowtime= time.time()
            img = dataset(video)
            z = np.zeros((1, 128, 128, 3))
            angle = predict(predictor, img, z)
            origin = Image.open("003.jpg")

            tensor_img = origin.resize((224,224), Image.BILINEAR)   #######resize 256*256
            
            if tensor_img.mode != 'RGB':
                tensor_img = tensor_img.convert('RGB')
            tensor_img = np.array(tensor_img).astype('float32').transpose((2, 0, 1))  # HWC to CHW
            tensor_img -= 127.5
            tensor_img *= 0.007843
            tensor_img = tensor_img[np.newaxis, :]

            
            tensor = pm.PaddleTensor()
            tensor.dtype =pm.PaddleDType.FLOAT32
            tensor.shape  = (1,3,224,224)
            tensor.data = pm.PaddleBuf(tensor_img)
            paddle_data_feeds1 = [tensor]
            count+=1
            outputs1 = predictor1.Run(paddle_data_feeds1)

            assert len(outputs1) == 1, 'error numbers of tensor returned from Predictor.Run function !!!'
            bboxes = np.array(outputs1[0], copy = False)
            print("bboxes.shape",bboxes.shape)

            t_labels = []
            t_scores = []
            t_boxes = []
            center_x = []
            center_y = []
            
            if len(bboxes.shape) == 1 :
                print("No object found in video")
                STATE_value =False
            else:
                STATE_value =False
                labels = bboxes[:, 0].astype('int32')
                scores = bboxes[:, 1].astype('float32')
                boxes = bboxes[:, 2:].astype('float32') 
                list_s = list(scores)
                max_score = max(list_s)
                print(labels, max_score)
                index = list_s.index(max_score)
    
                if max_score > 0.65:
                    max_box = boxes[index]
                    max_label = int(labels[index])
                    
                    if cout_i == 0:
                        last_label = max_label
                        cout_i += 1
                    else:
                        if max_label == last_label:
                            cout_i += 1
                        else:
                            cout_i = 0
                    print(max_label)

                            
                    if cout_i == 2:
                        cout_i = 0
                        for index in range(len(max_box)):
                            if index == 0 or index == 2:
                                max_box[index] = (max_box[index]/608) * 320
                            elif index == 1 or index == 3:
                                max_box[index] = (max_box[index]/608) * 240
                        center_x= int((max_box[0] + max_box[2]) / 2)
                        center_y = int((max_box[1] + max_box[3]) / 2)
                        print(center_x, center_y)
                        if max_label == 0:#限速              
                            if(center_y >= 190):
                                vels = 1502
                            else:
                                print('ok')
                                if (center_x - 160) >= 10 or (center_x - 160)<= -10:
                                    angle1 = center_x - 160
                                    angle = -angle1/250.0 + angle
                                    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~angle1:',angle)
                        if max_label == 1:#取消限速
                            if(center_y >= 185):
                                vels = 1510
                                flag1 = 1
                        elif max_label == 2:#左转
                            if(center_y >= 268):
                                vels = 1510
                        elif max_label == 3:#停车
                            if(center_y >= 220):
                                vels = 1500
                        elif max_label == 4 and flag == 0:#人行道
                            if(center_y >= 130):
                                lib.send_cmd(1500, 1500)
                                time.sleep(1.2)
                                flag = 1
                        elif max_label == 5 and flag1 == 1:#红车
                            if(center_y >= 85):
#                                lib.send_cmd(1500, 1500)
#                                time.sleep(0.2)
                                lib.send_cmd(1506, 1500)
                                print(1506)
                                time.sleep(0.4)
                                lib.send_cmd(1500, 1500)
                                flag1 = 2
                        elif max_label == 6:
                            if flag2 == 0:#红线
                                if(center_y >= 170):
                                    lib.send_cmd(1501, 1500)
                                    time.sleep(1.2)
                                    flag2 = 1
                            else:
                                for i in range(len(labels)):
                                    if int(labels[i]) == 3 and scores[i] > 0.6:
                                        for index in range(len(boxes[i])):
                                            if index == 0 or index == 2:
                                                boxes[i][index] = (boxes[i][index]/608) * 320
                                            elif index == 1 or index == 3:
                                                boxes[i][index] = (boxes[i][index]/608) * 240
                                        center_x= int((boxes[i][0] + boxes[i][2]) / 2)
                                        center_y = int((boxes[i][1] + boxes[i][3]) / 2)
                                        if center_y > 180:
                                            stop = 1
                            if stop == 1 and flag2 == 1 and center_y >= 200: 
                                print('OK')
                                while 1:
                                    lib.send_cmd(1508, 1500)
                   
            ################################################################################################
            vel = int(vels)

            a = int(angle * 2000 + 500)
            print('speed:',vel)
            print('angel:',a)
            
            lib.send_cmd(vel, a)
            print(cout)
            cout=cout+1
            print("cost:",time.time()-nowtime)
    '''
    except:
        print('error')
    finally:
        lib.send_cmd(1500, 1500)'''
