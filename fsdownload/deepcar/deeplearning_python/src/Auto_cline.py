# -*- coding:utf-8 -*-

"""
Three process runs
A process captures images
A process run lane line model, get the Angle
A process fetches TCP data to get the target category
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
import socket

path = os.path.split(os.path.realpath(__file__))[0] + "/.."
opts, args = getopt.getopt(argv[1:], '-hH', ['save_path=', 'vels=', 'camera='])

camera = multiprocessing.Array("b", range(50))
camera.value = "/dev/video2"
save_path = 'model_infer'
vels = 1535
crop_size = 128

lock = multiprocessing.Manager().Lock()
lock2 = multiprocessing.Manager().Lock()

for opt_name, opt_value in opts:
    #  The terminal gets the relevant parameters
    #  终端获取相关参数

    if opt_name in ('-h', '-H'):
        print("python3 Auto_Driver.py --save_path=%s --obj_path=%s --vels=%d --camera=%s " % (
        save_path, vels, camera))
        exit()

    if opt_name in ('--save_path'):
        save_path = opt_value


    if opt_name in ('--vels'):
        vels = int(opt_value)

    if opt_name in ('--camera'):
        camera = opt_value


def save_image_process(lock, camera, state):
    #  Capture picture process, and save
    #  采集图片进程，并保存

    global path
    video = v4l2capture.Video_device(camera.value)
    video.set_format(320, 240, fourcc='MJPG')
    video.create_buffers(50)
    video.queue_all_buffers()
    video.start()
    while 1:
        while state.value == True:
            now_time = time.time()
            select.select((video,), (), ())
            image_data = video.read_and_queue()
            frame1 = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            '''SAVE IMAGE TO LOCAL'''
            lock.acquire()
            cv2.imwrite(path + "/1.jpg", frame1)
            state.value = False
            lock.release()
            print("get  image process cost :", 1 / float(time.time() - now_time))


def dataset(frame):
    #  Preprocessing of images taken on the roadway
    #  车道上图片预处理

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
    print("image____shape:", img.shape)
    return img


def load_model():
    #  Load lane line model and model parameters
    #  加载车道线模型和模型参数

    valid_places = (
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
    #  Lane line picture forecast Angle
    #  车道线图片预测角度

    img = image

    i = predictor.get_input(0)
    i.resize((1, 3, 128, 128))
    z[0, 0:img.shape[1], 0:img.shape[2] + 0, 0:img.shape[3]] = img
    z = z.reshape(1, 3, 128, 128)
    frame1 = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite("test.jpg", frame1)
    i.set_data(z)

    predictor.run()
    out = predictor.get_output(0)
    score = out.data()[0][0]
    print(out.data()[0])
    return score


def control_car_process(lock, vels, state, lock2):
    #  Control car process
    #  控制车进程

    global path, save_path
    save_path = path + "/model/" + save_path
    cout = 0
    predictor = load_model()

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
            frame = cv2.imread(path + "/1.jpg")
            state.value = True
            lock.release()
            img = dataset(frame)
            z = np.zeros((1, 128, 128, 3))
            angle = predict(predictor, img, z)
            lock2.acquire()
            vel = int(vels.value)
            lock2.release()
            # a = int(angle*2000 + 500)
            a = int(angle * 3000 + 0)
            lib.send_cmd(vel, a)
            print(cout)
            cout = cout + 1


def get_data_process(vels, lock2):
    #  TCP fetch data
    #  TCP获取数据

    tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_client.connect(('127.0.0.1', 1234))
    print('Connection successful')

    while 1:
        recv_ = tcp_client.recv(1024)
        recv_data = recv_.decode('utf-8')
        print("OK")
        if (len(recv_data) > 20):
            print(len(recv_data))
            continue
        if (recv_data.find('null') == -1):
            lock2.acquire()
            vels.value = int(recv_data)
            lock2.release()
            print(vels.value )


if __name__ == "__main__":

    lib_path = path + "/lib" + "/libart_driver.so"
    so = cdll.LoadLibrary
    lib = so(lib_path)

    STATE = multiprocessing.Value("i", True)
    vels = multiprocessing.Value("i", vels)

    try:
        process_image = multiprocessing.Process(target=save_image_process, args=(lock, camera, STATE),
                                                name='process_image')
        process_car = multiprocessing.Process(target=control_car_process, args=(lock, vels, STATE, lock2),
                                              name='process_car')
        process_tcp = multiprocessing.Process(target=get_data_process, args=(vels, lock2),
                                              name='process_tcp')
        process_image.start()
        process_car.start()
        process_tcp.start()
        while 1:
            {}

    except:
        lib.send_cmd(1500, 1500)
        print('error')
    finally:
        lib.send_cmd(1500, 1500)
        print('finally')
