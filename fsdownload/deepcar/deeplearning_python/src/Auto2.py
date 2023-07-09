# -*- coding:utf-8 -*-
"""
get image from camera:/dev/video2  424*240

deal 128 *128

get the angle
"""

import v4l2capture
import sys, select, termios, tty
import os
import time
import threading
from ctypes import *
import numpy as np
import cv2 
from sys import argv


from paddlelite import *
import paddlemobile as pm
from PIL import Image
import getopt
import multiprocessing

#script,vels,save_path= argv

path = os.path.split(os.path.realpath(__file__))[0]+"/.."
opts,args = getopt.getopt(argv[1:],'-hH',['save_path=','vels=','camera='])

camera = multiprocessing.Array("b",range(50))
camera.value = "/dev/video2"
save_path = 'model_infer'
obj_path = 'freeze_model4'
vels  = 1535
crop_size = 128

lock = multiprocessing.Manager().Lock()

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


def save_image_process(lock, camera, state):

    global path
    video = v4l2capture.Video_device(camera.value)
    video.set_format(424, 240, fourcc='MJPG')
    video.create_buffers(50)
    video.queue_all_buffers()
    video.start()
    while 1:
        while (state.value == True):
            
            nowtime1 = time.time()
            select.select((video,), (), ())
            image_data = video.read_and_queue()
            frame1 = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            '''SAVE IMAGE TO LOCAL'''
            lock.acquire()
            cv2.imwrite(path + "/1.jpg", frame1)
            state.value = False
            lock.release()
            print("get  image process cost :", 1 / float(time.time() - nowtime1))



def dataset(frame):
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
    valid_places =   (
		Place(TargetType.kFPGA, PrecisionType.kFP16, DataLayoutType.kNHWC),
		Place(TargetType.kHost, PrecisionType.kFloat),
		Place(TargetType.kARM, PrecisionType.kFloat),
	)
    config = CxxConfig()
    model = save_path
    model_dir = model
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
    cv2.imwrite("zzzzz_test.jpg", frame1)
    i.set_data(z)

    predictor.run()
    out = predictor.get_output(0)
    score = out.data()[0][0]
    print(out.data()[0])
    return score


def load_model_detect(obj_path):

    path = obj_path
    model_dir = path
    print(model_dir)
    pm_config1 = pm.PaddleMobileConfig()
    pm_config1.precision = pm.PaddleMobileConfig.Precision.FP32  ######ok
    pm_config1.device = pm.PaddleMobileConfig.Device.kFPGA  ######ok

    pm_config1.model_dir = model_dir
    pm_config1.thread_num = 4
    predictor1 = pm.CreatePaddlePredictor(pm_config1)

    return predictor1

def control_car_process(lock,vels,state):
    global path, save_path, obj_path
    save_path = path + "/model/" + save_path
    obj_path = path + "/model/" + obj_path
    cout = 0
    predictor = load_model()
    predictor1 = load_model_detect(obj_path)
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
            lock.acquire()
            frame = cv2.imread(path+"/1.jpg")
            state.value =True
            lock.release()
            img = dataset(frame)
            origin = Image.fromarray(frame)
            z = np.zeros((1, 128, 128, 3))
            angle = predict(predictor, img, z)

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
            batch_outputs = predictor1.Run(paddle_data_feeds1)
            bboxes = np.array(batch_outputs[0])  # labels scores boxs
            if len(bboxes.shape) == 1:
                print("No object found in video")
            else:
                labels = bboxes[:, 0].astype('int')
                scores = bboxes[:, 1].astype('float')
                boxes = bboxes[:, 2:].astype('float')

                list_s = list(scores)
                max_score = max(list_s)
                index = list_s.index(max_score)

                max_box = boxes[index]
                max_label = int(labels[index])
                print(max_label)
#                center_x= int((max_box[0] + max_box[2]) / 2)
#                center_y = int((max_box[1] + max_box[3]) / 2)
                if max_label == 0:
                    vels = 1535
                elif max_label == 1:
                    vels = 1560
                elif max_label == 2:
                    vels = 1500
                elif max_label == 3:
                    vels = 1560
                elif max_label == 4:
                    vels = 1560
                elif max_label == 5:
                    vels = 1500
                else:
                    vels = 1560
            # a = int(angle*2000 + 500)
            vel = int(vels)
            print(vel)
            a = int(angle * 3000 + 0)
            lib.send_cmd(vel, a)
            print(cout)
            cout = cout + 1




if __name__ == "__main__":
    STATE = multiprocessing.Value("i", True)
    try:
        process_image = multiprocessing.Process(target=save_image_process, args=(lock, camera, STATE))
        process_car = multiprocessing.Process(target=control_car_process, args=(lock, vels, STATE))
        process_image.start()
        process_car.start()

        while 1:
            {}

    except:
        print('error')
    finally:
        lib.send_cmd(1500, 1500)
