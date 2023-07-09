import config
import serial_port
from serial_port import serial_connection
from cart import *
import sys
import datetime
import time
from driver import Driver

driver = Driver()
serial = serial_connection
flag1 = 0
flag2 = 0
#count = 0

count = 0

flag_camera1 = 0
flag_camera2 = 0
flag_stop = 0
flag_servo1 = 0

Flag_d = 0

flag_target_count = 1

flag_servo_angle = 0

flag_dis = 0
i = 0
recv_data = "background"

'''
def Integration(city_flag):
    if city_flag == 1:
        return 1
    if city_flag == 2:
        return 2
    if city_flag == 3:
        return 3
    if city_flag == 0:
        return 0
    if city_flag == 4:
        return 4
    if city_flag == 5:
        return 5
    else:
        return -1
'''


# def recv_tar(recv1):
#    global flag_t1, flag_t2, flag_t3, i, recv_data
#    if recv1 != "background":
#        i = + 1
#    if i == 1:
#        flag_t1 = 1
#    if i == 2 and recv1 == recv_data:
#        flag_t2 = 1
#    if i == 3 and recv1 == recv_data:
#        flag_t3 = 1
#    if i == 4 and recv1 == recv_data:
#        flag_t1 = 0
#        flag_t2 = 0
#        flag_t3 = 0
#        i = 0
#        return recv_data
#    if recv1 != recv_data and i < 4:
#        flag_t1 = 0
#        flag_t2 = 0
#        flag_t3 = 0
#        i = 0
#    recv_data = recv1
#    return "background"
def recv_tar(recv1):
    global i, recv_data
    if recv1 != "background":
        i = i + 1
        recv_data = recv1
    if i == 2 and recv1 == recv_data:
        i = 1
        recv_data = "background"
        return recv1
    return "background"


def task_count(taskflag):
    global count, flag_target_count
    if taskflag == 1 or taskflag == 1.1:
        count = count+1  # 这是滞后检测防止摄像头误转的不能保证提前检测！
        return count  # 计数检测只能等目标检测准确度上去了才能用，否则误判一下就会全盘崩，不建议用这种方法，虽然这种方法会减少比赛用时（迫不得已）

    if taskflag == 1.2:
        flag_target_count = flag_target_count+1
        count = count+1  # 这是滞后检测防止摄像头误转的不能保证提前检测！
        return count
        #  new
    if taskflag == 1.3:
        count = count+2  
        return count
        #  new
    
    return count


def return_flag_target_count():
    global flag_target_count
    return flag_target_count


def return_flag_servo_angle():
    global flag_servo_angle
    return flag_servo_angle


def camera_control_count(count):  # 方案二
    global flag_servo_angle
#    global Flag_d, flag_servo_angle
##    if count == 5:
##        Flag_d = count
##    if count == 7:
#    if count == None or count == -1:
#        return
    
#    Flag_d = count
    print("count=", count)
    if count == 1 or count == 7:
        flag_servo_angle = 180
        send_data_control_servo1 = bytes.fromhex('77 68 06 00 02 0B 02 30 B4 0A')  # 180度  转到右边
        serial.write(send_data_control_servo1)
        
#    if count == 6 or count == 8:
    if count == 5 or count == 8 or count == 6:
        flag_servo_angle = 0
        send_data_control_servo1 = bytes.fromhex('77 68 06 00 02 0B 02 30 00 0A')  # 0度    转到左边
        serial.write(send_data_control_servo1)
        



def camera_servo_control(recv, args):  # 方案一    1舵机   有问题待修改!!!!!!!
    global flag_camera1, flag_camera2, flag_servo1
    send_data_control_servo1 = bytes.fromhex('77 68 06 00 02 0B 02 30 00 0A')  # 0度
    serial.write(send_data_control_servo1)
    if recv == "fortress":
        flag_camera1 = 1
    if args == "daijun" or args == "dunhuang" or args == "dingxiangjun":
        flag_camera2 = 1
        if flag_camera1 == 1 and flag_camera2 == 1:
            flag_servo1 = 0
        return
    else:
        if flag_servo1 != 1:
            send_data_control_servo1 = bytes.fromhex('77 68 06 00 02 0B 02 30 B4 0A')  # 180度
            serial.write(send_data_control_servo1)
            flag_servo1 = 1
            return
    '''
    if recv == "barracks":
        send_data_control_servo1 = bytes.fromhex('77 68 06 00 02 0B 02 30 00 0A')  # 0度
        serial.write(send_data_control_servo1)
    '''
    if recv == "fenglangjuxu":
        flag_camera1 = 1
    if args == "trophies":
        flag_camera2 = 1
        if flag_camera1 == 1 and flag_camera2 == 1:
            flag_servo1 = 0
        return
    else:
        if flag_servo1 != 2:
            send_data_control_servo1 = bytes.fromhex('77 68 06 00 02 0B 02 30 B4 0A')  # 180度
            serial.write(send_data_control_servo1)
            flag_servo1 = 2
            return
    if recv == "target":
        flag_camera1 = 1
    if args == "target":
        flag_camera2 = 1
        if flag_camera1 == 1 and flag_camera2 == 1:
            flag_servo1 = 0
            return
    else:
        if flag_servo1 != 3:
            send_data_control_servo1 = bytes.fromhex('77 68 06 00 02 0B 02 30 B4 0A')  # 180度
            serial.write(send_data_control_servo1)
            flag_servo1 = 3
            return


def task_init():
    send_data_control_servo2 = bytes.fromhex('77 68 06 00 02 0B 03 20 14 0A')
    serial.write(send_data_control_servo2)
    send_data_control_servo3 = bytes.fromhex('77 68 06 00 02 0B 04 20 5F 0A')  # 30
    serial.write(send_data_control_servo3)
    send_data_control_servo4 = bytes.fromhex('77 68 06 00 02 0B 05 20 20 0A')
  # 20
    serial.write(send_data_control_servo4)
    send_data_control_servo_city1 = bytes.fromhex('77 68 06 00 02 0B 06 50 B4 0A')  # B490度
  
    serial.write(send_data_control_servo_city1)
    send_data_control_servo_city2 = bytes.fromhex('77 68 06 00 02 0B 07 50 B4 0A')
    serial.write(send_data_control_servo_city2)
    send_data_control_servo_city3 = bytes.fromhex('77 68 06 00 02 0B 08 50 00 0A')
    serial.write(send_data_control_servo_city3)
    send_data_control_motor = bytes.fromhex('77 68 06 00 02 0C 02 02 00 0A')  # 0
    serial.write(send_data_control_motor)
    send_data_control_servo1 = bytes.fromhex('77 68 06 00 02 0B 02 30 00 0A')  #   00 0度   B4 180度
     
    serial.write(send_data_control_servo1)
    send_data_control_white = bytes.fromhex('77 68 08 00 02 3B 02 01 00 00 00 0A')
    serial.write(send_data_control_white)
    print("Task initialization completed......\n")


def returnflagtask(flag_):
    if flag_ == 1 or flag_ == 1.1 or flag_ == 1.2 or flag_ == 1.3:
        return 1
    else:
        return 0

def change_posture_me(dis):
    speed = -30
    timer =1
    car = Cart()
    if dis <0.1 and dis > 0:
        timer = dis *timer *10
    if dis > -0.1 and dis < 0:
        timer = -dis *timer *10
    if dis > 0.1:
        timer = dis*timer
    if dis<-0.1:
        timer = -dis*timer 
    speed = speed * dis * 5
    timer = timer*100//100
#    if speed < 10 and speed > 0:
#        speed = 10
#    if speed >20:
#        speed = 20
#    if speed < -20:
#         speed = -20
#    if speed > -10 and speed < 0:
#        speed = -10
    car.move([speed, speed, speed, speed])
    print("speed=", speed)
    print("dis=", dis)
    print("timer=", timer)
    time.sleep(timer)
    driver.stop()
#
#def run_task(recv, args, dis):

def run_task(recv, args):

    global flag1, flag2, flag_stop
    global Flag_d
    global count
    global flag_dis
    car = Cart()
    #    if recv == "background" and args == "background":   # 同时与检测！！！
    #        flag_stop = 0
    #        return -3
    #    if recv == "background" or args == "background":    # 不同时或检测！！！
    #        flag_stop = 0
    #        return -4
    #    if recv is None or args is None:                    # 异常检测归为不同时异或检测！！！
    #        flag_stop = 0
    #        return -4

    print("Flag_d=", Flag_d)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#    if recv == "soldier" and Flag_d == 2:  # 粮草  1舵机
    if recv == "soldier":  # 粮草  1舵机

        flag1 = 1
    if args == "forage" and flag1 == 1:
#    if args == "forage":
        #        if flag_stop == 0:
#        time.sleep(2)
        driver.stop()
        send_data_control_servo2 = bytes.fromhex('77 68 06 00 02 0B 03 14 70 0A')
  # 6E
        serial.write(send_data_control_servo2)
        time.sleep(3)
        send_data_control_servo2 = bytes.fromhex('77 68 06 00 02 0B 03 20 14 0A')
  # 37
        serial.write(send_data_control_servo2)
#        time.sleep(0.5)
#        send_data_control_red = bytes.fromhex('77 68 08 00 02 3B 02 00 FF C0 CB 0A')
#        serial.write(send_data_control_red)
#        time.sleep(2)
#        send_data_control_white = bytes.fromhex('77 68 08 00 02 3B 02 00 00 00 00 0A')
#        serial.write(send_data_control_white)

        flag1 = 0
        flag2 = 0
        #            flag_stop = 1
        Flag_d = 0
        return 1
#    if recv == "fenglangjuxu"and Flag_d == 1:  # 战利品 2舵机
    if recv == "fenglangjuxu":  # 战利品 2舵机

        flag1 = 2
    if args == "trophies":
        flag2 = 5
        if flag1 == 2 and flag2 == 5:
            #            if flag_stop == 0:
            driver.stop()
            time.sleep(0.5)
#            send_data_control_blue = bytes.fromhex('77 68 08 00 02 3B 02 00 00 00 FF 0A')
#            serial.write(send_data_control_blue)
#            time.sleep(2)
#            send_data_control_white = bytes.fromhex('77 68 08 00 02 3B 02 00 00 00 00 0A')
#            serial.write(send_data_control_white)

#
#            send_data_control_servo3 = bytes.fromhex('77 68 06 00 02 0B 05 20 60 0A')  # 105
#            serial.write(send_data_control_servo3)
#            time.sleep(1)
#
#            send_data_control_servo4 = bytes.fromhex('77 68 06 00 02 0B 05 20 20 0A')
#            serial.write(send_data_control_servo4)
#            time.sleep(1)
            send_data_control_servo3 = bytes.fromhex('77 68 06 00 02 0B 04 20 9B 0A')  # B4
            serial.write(send_data_control_servo3)
            time.sleep(1)
            send_data_control_servo3 = bytes.fromhex('77 68 06 00 02 0B 05 20 B4 0A')  # 30
            serial.write(send_data_control_servo3)   
            time.sleep(1)        
            send_data_control_servo3 = bytes.fromhex('77 68 06 00 02 0B 04 20 5F 0A')  # 105
            serial.write(send_data_control_servo3)
            time.sleep(1)
#            send_data_control_servo4 = bytes.fromhex('77 68 06 00 02 0B 05 20 08 0A')
#            serial.write(send_data_control_servo4)
            
#            time.sleep(1)
#            send_data_control_servo3 = bytes.fromhex('77 68 06 00 02 0B 04 20 1E 0A')  # 30
#            serial.write(send_data_control_servo3)
#            time.sleep(1)
            flag1 = 0
            flag2 = 0
            Flag_d = 2
            #                flag_stop = 2
            return 1
        return -1

    if recv == "fortress":  # 要塞  3舵机
        flag1 = 3
        print("flag1=", flag1)
    if args == "daijun":  # 代郡
        flag2 = 3.1
        print("flag2=", flag2)
        print("daijun is called....................................")
        
        if count == 8:
            driver.stop()
            car.move([-10, -10, -10, -10])
            time.sleep(2)
            print("this....................................")
        if count == 0:
            driver.stop()
            car.move([-10, -10, -10, -10])
            time.sleep(1)
            print("that....................................")


        if flag1 == 3 and flag2 == 3.1:
            
            driver.stop()
#            time.sleep(1)
#            time.sleep(1)
            print("daijun...................................................")
            send_data_control_servo_city1 = bytes.fromhex('77 68 06 00 02 0B 06 50 5A 0A')
            serial.write(send_data_control_servo_city1)
            time.sleep(2)
            send_data_control_green = bytes.fromhex('77 68 08 00 02 3B 02 00 00 FF 00 0A')
            serial.write(send_data_control_green)
            time.sleep(1)
            send_data_control_white = bytes.fromhex('77 68 08 00 02 3B 02 00 00 00 00 0A')
            serial.write(send_data_control_white)
            time.sleep(1)
            send_data_control_green = bytes.fromhex('77 68 08 00 02 3B 02 00 00 FF 00 0A')
            serial.write(send_data_control_green)
            time.sleep(1)
            send_data_control_white = bytes.fromhex('77 68 08 00 02 3B 02 00 00 00 00 0A')
            serial.write(send_data_control_white)
            time.sleep(1)
            send_data_control_green = bytes.fromhex('77 68 08 00 02 3B 02 00 00 FF 00 0A')
            serial.write(send_data_control_green)
            time.sleep(1)
            send_data_control_white = bytes.fromhex('77 68 08 00 02 3B 02 00 00 00 00 0A')
            serial.write(send_data_control_white)
            send_data_control_servo_city1 = bytes.fromhex('77 68 06 00 02 0B 06 50 B4 0A')
            serial.write(send_data_control_servo_city1)
            flag1 = 0
            flag2 = 0
            flag_stop = 3
            return 1
        return -1
    if args == "dunhuang":  # 敦煌
   
        flag2 = 3.2
        
        if count == 8:
            driver.stop()
            car.move([-10, -10, -10, -10])
            time.sleep(2)
            print("this....................................")
        if count == 0:
            driver.stop()
            car.move([-10, -10, -10, -10])
            time.sleep(1)
            print("that....................................")



        if flag1 == 3 and flag2 == 3.2:
            #            if flag_stop == 0:
            
            driver.stop()
#            time.sleep(1)
            send_data_control_servo_city2 = bytes.fromhex('77 68 06 00 02 0B 08 50 5A 0A')
            serial.write(send_data_control_servo_city2)
            time.sleep(2)
            send_data_control_green = bytes.fromhex('77 68 08 00 02 3B 02 00 00 FF 00 0A')
            serial.write(send_data_control_green)
            time.sleep(1)
            send_data_control_white = bytes.fromhex('77 68 08 00 02 3B 02 00 00 00 00 0A')
            serial.write(send_data_control_white)
            time.sleep(1)
            send_data_control_green = bytes.fromhex('77 68 08 00 02 3B 02 00 00 FF 00 0A')
            serial.write(send_data_control_green)
            time.sleep(1)
            send_data_control_white = bytes.fromhex('77 68 08 00 02 3B 02 00 00 00 00 0A')
            serial.write(send_data_control_white)
            time.sleep(1)
            send_data_control_green = bytes.fromhex('77 68 08 00 02 3B 02 00 00 FF 00 0A')
            serial.write(send_data_control_green)
            time.sleep(1)
            send_data_control_white = bytes.fromhex('77 68 08 00 02 3B 02 00 00 00 00 0A')
            serial.write(send_data_control_white)
            send_data_control_servo_city2 = bytes.fromhex('77 68 06 00 02 0B 08 50 00 0A')
            serial.write(send_data_control_servo_city2)
            flag1 = 0
            flag2 = 0
            #                flag_stop = 3
            return 1
        return -1
    if args == "dingxiangjun":  # 定襄郡
        flag2 = 3.3
        
        if count == 8:
            driver.stop()
            car.move([-10, -10, -10, -10])
            time.sleep(2)
            print("this....................................")
        if count == 0:
            driver.stop()
            car.move([-10, -10, -10, -10])
            time.sleep(1)
            print("that....................................")


        if flag1 == 3 and flag2 == 3.3:
            driver.stop()
            print("dinxiangjun is called....................................")
#            time.sleep(1)
            print("dingxiangjun...................................................")
            send_data_control_servo_city3 = bytes.fromhex('77 68 06 00 02 0B 07 50 5A 0A')
            serial.write(send_data_control_servo_city3)
            time.sleep(2)
            send_data_control_green = bytes.fromhex('77 68 08 00 02 3B 02 00 00 FF 00 0A')
            serial.write(send_data_control_green)
            time.sleep(1)
            send_data_control_white = bytes.fromhex('77 68 08 00 02 3B 02 00 00 00 00 0A')
            serial.write(send_data_control_white)
            time.sleep(1)
            send_data_control_green = bytes.fromhex('77 68 08 00 02 3B 02 00 00 FF 00 0A')
            serial.write(send_data_control_green)
            time.sleep(1)
            send_data_control_white = bytes.fromhex('77 68 08 00 02 3B 02 00 00 00 00 0A')
            serial.write(send_data_control_white)
            time.sleep(1)
            send_data_control_green = bytes.fromhex('77 68 08 00 02 3B 02 00 00 FF 00 0A')
            serial.write(send_data_control_green)
            time.sleep(1)
            send_data_control_white = bytes.fromhex('77 68 08 00 02 3B 02 00 00 00 00 0A')
            serial.write(send_data_control_white)
            send_data_control_servo_city3 = bytes.fromhex('77 68 06 00 02 0B 07 50 B4 0A')
            serial.write(send_data_control_servo_city3)
            flag1 = 0
            flag2 = 0
            #                flag_stop = 3
            return 1
        return -1
#    if recv == "barracks" and Flag_d == 0:  # 侧方停车
    if recv == "barracks":
        #        if flag_stop == 0:
        #        	return
        driver.stop()
        time.sleep(1)
#        1

        car.move([20, 50, 20, 50])
        
        time.sleep(0.6)
   # 0.5
        driver.stop()
        car.move([20, 20, 20, 20])
        time.sleep(2.1)    #2
        driver.stop()
        car.move([60, 20, 60,20])
        time.sleep(0.5)   # 0.5
        driver.stop()



        
        send_data_control_red = bytes.fromhex('77 68 08 00 02 3B 02 00 FF 00 00 0A')
        serial.write(send_data_control_red)
        time.sleep(1)
        send_data_control_white = bytes.fromhex('77 68 08 00 02 3B 02 00 00 00 00 0A')
        serial.write(send_data_control_white)
        time.sleep(1)
        send_data_control_red = bytes.fromhex('77 68 08 00 02 3B 02 00 FF 00 00 0A')
        serial.write(send_data_control_red)
        time.sleep(1)
        send_data_control_white = bytes.fromhex('77 68 08 00 02 3B 02 00 00 00 00 0A')
        serial.write(send_data_control_white)
        time.sleep(1)
        send_data_control_red = bytes.fromhex('77 68 08 00 02 3B 02 00 FF 00 00 0A')
        serial.write(send_data_control_red)
        time.sleep(1)
        send_data_control_white = bytes.fromhex('77 68 08 00 02 3B 02 00 00 00 00 0A')
        serial.write(send_data_control_white)

        car.move([-40, -10, -40, -10])
        time.sleep(0.5)     # 0.9
        car.move([-20, -20, -20, -20])
        time.sleep(0.6)     # 0.8
        car.move([40, -20, 40,-20])
        time.sleep(0.7)   # 0.5
#        1
        
#        car.move([60, 10, 60, 10])     # -10
#        time.sleep(0.5)
#        driver.stop()
        car.move([20, 20, 20, 20])     # -10
        time.sleep(0.5)
#        time.sleep(1.3)  # 1.2
#        car.move([10, 30, 10, 30])     # -10
#        time.sleep(0.5)    # 0.4
#        driver.stop()
#        
#        car.move([-15, -15, -15, -15])     # -10
#        time.sleep(1.6)    #    1.2
#        driver.stop()
#        
#        car.move([30, 10, 30, 10])     # -10
#        time.sleep(0.1)  # 0.1
        car.move([15, 40, 15, 40])     # -10
        time.sleep(0.4)
        driver.stop()


        
        flag1 = 0
        flag2 = 0
        flag_stop = 4
        Flag_d = 1
        
        if count <= 4:          # zhis 
            count = 4

#        if count <= 3:          # zhis 
#            count = 3
            return 1.3        # 1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        else:
            return 1
    if recv == "target":  # 打靶
        flag1 = 5
        
        
            
    if args == "target":
        flag2 = 4
        # new
#        if flag_dis == 0:
##            change_posture_me(dis)
#            send_data_control_green = bytes.fromhex('77 68 08 00 02 3B 02 00 00 AA 00 0A')
#            serial.write(send_data_control_green)
#            driver.stop()
#            flag_dis = 1
#            print("Target is called and funcation is called....................................")
#            return 0
        # new
        print("Target is called....................................")
        #        print("....................................")
        #        print("....................................")
        #        print("....................................")
        #        print("....................................")
        #        print("....................................")
        #        print("....................................")
        #        print("....................................")
        #        print("....................................")
        #        print("....................................")
        #        print("....................................")
        #        print("....................................")
        #        print("....................................")
        #        print("....................................")
        #        print("....................................")
        #        print("....................................")
        #        print("....................................")
        #        print("....................................")
        #        print("....................................")
        #        print("....................................")
        #        print("....................................")
        #        print("....................................")
        #        print("....................................")
        #        print("....................................")
        #        print("....................................")
        if flag1 == 5 and flag2 == 4:  # and flag_stop == 0:

            speed1 = 20  # 正转
            speed2 = -15  # 反转
            driver.stop()
            time.sleep(1)
            send_data_control_motor = bytes.fromhex('77 68 06 00 02 0C 02 02') + speed1.to_bytes(1, byteorder='big',
                                                                                                 signed=True) + bytes.fromhex(
                '0A')  # 正转  # EC
            serial.write(send_data_control_motor)
            time.sleep(1)
            print("Target is ok..............................................")
            send_data_control_motor = bytes.fromhex('77 68 06 00 02 0C 02 02') + speed2.to_bytes(1, byteorder='big',
                                                                                                 signed=True) + bytes.fromhex(
                '0A')  # 反转  # EC
            serial.write(send_data_control_motor)
            time.sleep(1)
#            send_data_control_motor = bytes.fromhex('77 68 06 00 02 0C 02 02') + speed1.to_bytes(1, byteorder='big',
#                                                                                                 signed=True) + bytes.fromhex(
#                '0A')  # 正转  # EC
#            serial.write(send_data_control_motor)
#            time.sleep(1)
#            send_data_control_motor = bytes.fromhex('77 68 06 00 02 0C 02 02') + speed2.to_bytes(1, byteorder='big',
#                                                                                                 signed=True) + bytes.fromhex(
#                '0A')  # 正转  # EC
#            serial.write(send_data_control_motor)
            send_data_control_motor = bytes.fromhex('77 68 06 00 02 0C 02 02 00 0A')  # 0
            serial.write(send_data_control_motor)
            flag1 = 0
            flag2 = 0
            flag_stop = 5
            flag_dis = 0
            return 1.2    # 1.2!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! old
        return -1


'''
返回值的含义： 0是满足第一层条件（地标）但不满足第二层条件（侧标）
            -1是满足第一层条件（地标）满足第二层条件（侧标），但不满足第三层条件（两种标志位都成立）
            1是正常执行任务
            -2是出错
            -3是背景同时满足复位摄像头标志位
            -4是背景不同时满足（过了同时检测不满足则一定不同时满足）时和None
'''


def main():
    task_init()
    while True:
        recv1 = "background"
        recv2 = "background"
        recv = "target"
        camera_servo_control(recv1, recv2)
        t = run_task(recv1, recv2)
        task_count(t)


if __name__ == '__main__':
    main()
