#!/usr/bin/env python
# -*- coding: utf-8 -*-
# прорамма для следования квадрокоптера вдоль дороги
# обнаруживает дорогу как центр цветового пятна (метод моментов)
# ориентируется по центру дороги

########IMORTS#########
import rospy
import cv2
import time
import math
import numpy as np
import ros_numpy as rnp
from dronekit import connect, VehicleMode, LocationGlobalRelative, LocationGlobal
from pymavlink import mavutil
from array import array
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from std_msgs.msg import Int16  # For error/angle plot publishing
from sensor_msgs.msg import Image
from simple_pid import PID
import matplotlib.pyplot as plt

# from drone_control import *

#######VARIABLES########
vehicle = connect('tcp:127.0.0.1:5763',wait_ready=True) #установление соединения с виртуальным дроном
vehicle.parameters['PLND_ENABLED']=1 #PLND = Precision landing, включаем.
vehicle.parameters['PLND_TYPE']=1 #companion computer (type of Precision landing) - вместе с компьютером.
#0: полагаемся только на сенсоры. В данном случае, только с камеры.
#1: Используем фильтр Калмана. Может привести к сомнительным результатам.
vehicle.parameters['PLND_EST_TYPE']=0
vehicle.parameters['LAND_SPEED']=30 ##скорость снижения cms/s

########################
#Издатель (объект, который публикует сообщения с камеры - преобразованное изображение)
#Создаём топик /camera/color/image_new, тип - изображение, размер буфера - 10 сбщ
newimg_pub = rospy.Publisher('/camera/color/image_new', Image, queue_size=10)
pub_vel = rospy.Publisher('iris_drone/cmd_vel', Twist, queue_size=1)
pub_error = rospy.Publisher('error', Int16, queue_size=10)
pub_angle = rospy.Publisher('angle', Int16, queue_size=10)

#горизонтальное и вертикальное разрешения рос газибо камеры в пикселях
horizontal_res = 1280 #640
vertical_res = 720 #480

velocity=1 #m/s
takeoff_height=15 #m
on_height = 0.0 #время достижения высоты

#####
time_last=0
time_to_wait = .1 ##100 ms

####
line_side = 0
err_array = []
ang_array = []
time_array = []

x_control =0
ang_control =0

################FUNCTIONS###############
def arm_and_takeoff(targetHeight):
    global on_height
    while vehicle.is_armable !=True: #готовность к взлёту
        print('Waiting for vehicle to become armable')
        time.sleep(1)
    print('Vehicle is now armable')

    vehicle.mode = VehicleMode('GUIDED') #режим управляемого полета

    while vehicle.mode !='GUIDED': #ждём инициализацию управляемого полета
        print('Waiting for drone to enter GUIDED flight mode')
        time.sleep(1)
    print('Vehicle now in GUIDED mode. Have Fun!')

    vehicle.armed = True #армим дрон (запускаем вращение винтов)
    while vehicle.armed ==False:
        print('Waiting for vehicle to become armed.')
        time.sleep(1)
    print('Look out! Virtual props are spinning!')

    vehicle.simple_takeoff(targetHeight) #взлетаем до заданной высоты

    while True:
        print('Current Altitude: %d'%vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt >=.95*targetHeight:
            break
        time.sleep(1)
    print('Target altitude reached!')
    on_height = time.time()

    return None

##Send velocity command to drone
def send_local_ned_velocity(vx,vy,vz, duration=1):
    print('vx, vy, vz = ' +str(vx)+', '+str(vy)+', '+str(vz))
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,
        0,
        0,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        #MAV_FRAME_BODY_OFFSET_NED делает компоненты скорости относительно текущего курса транспортного средства.
        0b0000111111000111,
        0,
        0,
        0,
        vx,
        vy,
        vz,
        0,0,0,0,0)
    vehicle.send_mavlink(msg)
    vehicle.flush()

def condition_yaw(heading, line_side, relative=1):
    if relative:
        is_relative = 1 #yaw relative to direction of travel
    else:
        is_relative = 0 #yaw is an absolute angle
    # create the CONDITION_YAW command using command_long_encode()
    print('heading = ' +str(heading))
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_CONDITION_YAW, #command
        0, #confirmation
        heading,    # param 1, yaw in degrees
        0,          # param 2, yaw speed deg/s
        line_side,          # param 3, direction -1 ccw, 1 cw
        is_relative, # param 4, relative offset 1, absolute angle 0
        0, 0, 0)    # param 5 ~ 7 not used
    # send command to vehicle
    vehicle.send_mavlink(msg)

def preprocessing(frame):
    land_mask = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(land_mask, (21, 21), 0)
    #фильтр по значению серого
    low_gr = np.array([0, 0, 30])
    up_gr = np.array([240, 10, 100])
    land_mask = cv2.inRange(blur, low_gr, up_gr)
    return land_mask

def find_moments(np_data):
    land_mask = preprocessing(np_data)

    y_stp = np_data.shape[0]/2
    # разбиваем изображение на три части и находим моменты
    im_top = land_mask[0:int(y_stp/3), 0:horizontal_res]
    im_mid = land_mask[int(y_stp/3):(2*int(y_stp/3)), 0:horizontal_res]
    im_bot = land_mask[(2*int(y_stp/3)):int(y_stp), 0:horizontal_res]

    moments_bot = cv2.moments(im_bot,1)
    moments_mid = cv2.moments(im_mid,1)
    moments_top = cv2.moments(im_top,1)
    return moments_bot, moments_mid, moments_top


#Функция, выполняемая при поступлении нового сообщения для подписчика
def msg_receiver(message):
    global time_last, time_to_wait, velocity, on_height
    global line_side, x_control, ang_control
    #имитируем низкий FPS (~10). Выполняется, если дельта t > времени задержки
    if (time.time() - time_last > time_to_wait) and (vehicle.mode !='LAND'):
        np_data = rnp.numpify(message) ## перевод изображения в массив данных

        # середина изображения (точка для слежения)
        x_stp = np_data.shape[1]/2
        y_stp = np_data.shape[0]/2

        moments_bot, moments_mid, moments_top = find_moments(np_data)
        print('                ')
        print('#################')
        #центр кадра
        cv2.line(np_data, (int(x_stp), int(y_stp+50)), (int(x_stp), int(y_stp)-50), (0, 0, 255), 3)
        cv2.line(np_data, (int(x_stp)+50, int(y_stp)), (int(x_stp)-50, int(y_stp)), (0, 0, 255), 3)
        try:
            # находим центр для каждой части изображения и рисуем там круги
            x_bot, x_mid, x_top, x_norm = 0, 0, 0, 0
            y_bot, y_mid, y_top = 0, 0, 0

            if moments_bot['m00']>100:
                x_bot = int(moments_bot['m10']/moments_bot['m00'])
                y_bot = int(moments_bot['m01']/moments_bot['m00'])
                y_bot = y_bot + 2*int(y_stp/3)
                cv2.circle(np_data, (x_bot, y_bot), 10, (255,0,0), -1)
            if moments_mid['m00']>100:
                x_mid = int(moments_mid['m10']/moments_mid['m00'])
                y_mid = int(moments_mid['m01']/moments_mid['m00'])
                y_mid = y_mid + int(y_stp/3)
                cv2.circle(np_data, (x_mid, y_mid), 10, (255,0,0), -1)
            if moments_top['m00']>100:
                x_top = int(moments_top['m10']/moments_top['m00'])
                y_top = int(moments_top['m01']/moments_top['m00'])
                cv2.circle(np_data, (x_top, y_top), 10, (255,0,0), -1)

            # высчитываем центр верхей части изображения для сравнения
            x_norm = int(((y_top - y_bot)*(x_mid - x_bot)/(y_mid - y_bot)) + x_bot)
            cv2.circle(np_data, (x_norm, y_top), 10, (0,255,0), -1)
            delta = x_norm - x_top #не используется

            #рисуем вектор направления дороги
            cv2.line(np_data, (x_bot, y_bot), (x_norm, y_top), (0, 255, 0), 3)

            print('x_bot =  ', x_bot,' x_mid = ', x_mid,' x_top = ', x_top,' x_norm = ', x_norm)
            print('y_bot =  ', y_bot,' y_mid = ', y_mid,' y_top = ', y_top)
            print('delta = ', delta)

            # рассчитываем правильное значение угла
            if (x_mid-x_bot)==0:
                angle = 0.0
            elif (x_mid-x_bot)!=0:
                angle = math.atan((y_bot-y_mid)/(x_mid-x_bot))
                angle = math.degrees(angle)
                if angle<0:
                    angle = -1*(90+angle)
                else:
                    angle = 90 - angle

            angle = int(angle)
            print('angle: ', angle)
            ang_array.append(angle)#пополняем массив углов

            #считаем ошибку по х
            error = int(x_bot - x_stp)
            print('error: ', error)
            err_array.append(error)#пополняем массив ошибок
            # переводим в значения [-1; 1]
            normal_error = float(error) / x_stp #относительная ошибка
            print('normal_error: ', normal_error)

            # ПИД-регулятор
            pid_x = PID(1.5, 0.1, 1,setpoint=normal_error)
            pid_ang = PID(0.25, 0.5, 0.01,setpoint=angle)
            pid_x.setpoint = 0
            pid_ang.setpoint = 0
            x_control = pid_x(normal_error)
            ang_control = pid_ang(angle)

            # в какой стороне линия
            if angle > 0:
                line_side = 1  #линия справа (поварачивает направо)
            elif angle < 0:
                line_side = -1  #линия слева (поварачивает налево)

            #выводим значение угла
            cv2.putText(np_data, "Angle: " + str(angle), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)
            #выводим значение ошибки
            cv2.putText(np_data, "Error: " + str(error), (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        cv2.LINE_AA)

            #публикуем линейные и угловые скорости
            twist = Twist()
            twist.linear.x = velocity
            twist.linear.y = x_control
            twist.linear.z = 0
            twist.angular.x = 0
            twist.angular.y = 0
            twist.angular.z = ang_control
            pub_vel.publish(twist)

            #публикуем углы отклонения
            ang = Int16()
            ang.data = angle
            pub_angle.publish(ang)

            #публикуем ошибки
            err = Int16()
            err.data = error
            pub_error.publish(err)

            time_array.append(time.time()-on_height)


            #Производим обратное преобразование данных в изображение формата рос Image
            new_msg = rnp.msgify(Image, np_data,encoding='rgb8') #сообщение формата РОС, которое мы будем публиковать в топик /camera/color/image_new
            newimg_pub.publish(new_msg) #публикуем новое изображение
        except Exception as e: #обработчик ошибок
            print('Target likely not found') #Цель возможно не найдена
            print(e)
            #Производим обратное преобразование данных в изображение формата рос Image
            new_msg = rnp.msgify(Image, np_data,encoding='rgb8') #сообщение формата РОС, которое мы будем публиковать в топик /camera/color/image_new
            newimg_pub.publish(new_msg) #публикуем новое изображение

        time_last = time.time() #сохраняем последнее время прогона кода
    else:
        return None #если прошло меньше 0.1 сек, то не делаем ничего



def controller(pub_vel):
    global velocity, line_side, x_control, ang_control
    global on_height
    ang_control = abs(ang_control)

    send_local_ned_velocity(velocity, -x_control, 0)
    condition_yaw(ang_control, line_side)
    print('flight time = ', time.time()-on_height)
    if time.time() - on_height >= 120.00:
        print('time to landing', time.time())
        #активация режима посадки
        if vehicle.mode !='LAND':
            vehicle.mode = VehicleMode('LAND')
            errorPlot()
            while vehicle.mode !='LAND':
                time.sleep(1)
            print('We find land! Vehicle in LAND mode')

#функция вывода значений
def errorPlot():
    # выводим данные
    meanError = np.mean(err_array) #средняя ошибка
    stdError = np.std(err_array)#стандартная ошибка

    meanAngle = np.mean(ang_array)#средний угол
    stdAngle = np.std(ang_array)#стандартный угол

    err_abs=[]
    ang_abs=[]
    for i in err_array:
        err_abs.append(abs(err_array[i]))

    mean_abs_Error = np.mean(err_abs) #средняя ошибка по модулю
    std_abs_Error = np.std(err_abs)#стандартная ошибка по модулю

    for i in ang_array:
        ang_abs.append(abs(ang_array[i]))

    mean_abs_Angle = np.mean(ang_abs)#средний угол по модулю
    std_abs_Angle = np.std(ang_abs)#стандартный угол по модулю

    print('  ')
    print('-----------')
    print('meanError: ', meanError, 'stdError: ', stdError)
    print('meanAngle: ', meanAngle, 'stdAngle: ', stdAngle)
    print('        ')
    print('mean_abs_Error: ', mean_abs_Error, 'mean_abs_Angle: ', mean_abs_Angle)

    # строим графики
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)
    ax.plot(time_array, ang_array)
    ax.grid()
    #  Добавляем подписи к осям:
    ax.set_xlabel('t (s)')
    ax.set_ylabel('Angle_m (deg)')

    ax2.plot(time_array, err_array)
    ax2.grid()
    #  Добавляем подписи к осям:
    ax2.set_xlabel('t (s)')
    ax2.set_ylabel('Error_m (pixels)')
    plt.show()


#Функция создания подписчика
def subscriber():
    rospy.init_node('drone_node',anonymous=False) #название ноды - drone_node
    #подписываемся на топик  (название, тип сообщения, выполняемая функция)
    sub = rospy.Subscriber('/camera/color/image_raw', Image, msg_receiver)
    sub_vel = rospy.Subscriber('iris_drone/cmd_vel', Twist, controller)

    rospy.spin() #зацикливаем


if __name__=='__main__':
    try:
        #time.sleep(10)
        arm_and_takeoff(takeoff_height)
        time.sleep(1)

        subscriber()
    except rospy.ROSInterruptException:
        pass
