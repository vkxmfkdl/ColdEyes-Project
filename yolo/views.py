from django.shortcuts import render
from django.shortcuts import render_to_response
from django.shortcuts import redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse,StreamingHttpResponse


import numpy as np
import urllib
import json
import cv2
import os
import cv2
import time
import subprocess
from django.views.decorators import gzip
import scipy.misc as smp
from PIL import Image
# Create your views here.
######################
import queue


######################
#from __future__ import division
import torch
import torch.nn as nn

from torch.autograd import Variable
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random
import argparse
import pickle as pkl

#메일관련 패키지
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
######################
# 디비연동
from yolo.models import fireoffice
import pymysql
#######################
# 날씨 패키지
import forecastio

#######################
#경로값
import tensorflow as tf


#######################
label_list = []
cnt_length_queue = queue.Queue()
label_acc_queue = queue.Queue()
label_temp_queue =queue.Queue() # 임시 복사해놓는 큐
frames = 0
flag = 0
collision_flag = 0 # 위험물질 충돌 발생 여부 .
picture = 0


#######################
def main(request):
    reset_data()
    return render_to_response("main.html")

def videofile1(request):
    reset_data()
    return render_to_response("videofile1.html")

def videofile2(request):
    reset_data()
    return render_to_response("videofile2.html")

def yolosite(request):
    reset_data()
    return render_to_response("yolosite.html")

#############################################################################
def get_test_input(input_dim, CUDA):
    img = cv2.imread("imgs/messi.jpg")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()

    return img_

def prep_image(img, inp_dim):
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img):
    classes = load_classes('data/coco.names')
    colors = pkl.load(open("pallete", "rb"))
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())

    x_map=0
    y_map=0
    width=0
    height=0
    x_map=int(x[1].int())
    y_map=int(x[2].int())
    width=int(x[3].int())
    height=int(x[4].int())

    L_list = []
    cls = int(x[-1])
    print(cls)
    if cls >4 or cls < 1:
        L_list.append("null")
        L_list.append(x_map)
        L_list.append(y_map)
        L_list.append(width)
        L_list.append(height)
        return L_list

    label = "{0}".format(classes[cls])
    color = random.choice(colors)

    if width == 0 or height == 0:
        L_list.append("null")
        L_list.append(x_map)
        L_list.append(y_map)
        L_list.append(width)
        L_list.append(height)
        return L_list

    L_list.append(label)
    L_list.append(x_map)
    L_list.append(y_map)
    L_list.append(width)
    L_list.append(height)

    #박스 띄움
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    # 라벨이미지 띄움
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return L_list


def stream_yolo_ready():
    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "yolov3.weights"
    num_classes = 5
    confidence = 0.7
    nms_thesh = 0.5
    start = 0
    CUDA = torch.cuda.is_available()

    num_classes = 5
    bbox_attrs = 5 + num_classes

    model = Darknet(cfgfile)
    model.load_weights(weightsfile)

    model.net_info["height"] = 416
    inp_dim = int(model.net_info["height"])

    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    model.eval()

    videofile = 'video.avi'

    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), 'Cannot capture source'
    cap.set(3,416)
    cap.set(4,416)

    global frames
    global picture
    frames = 0

    start = time.time()
    while cap.isOpened():

        ret, frame = cap.read()
        if ret:

            img, orig_im, dim = prep_image(frame, inp_dim)
            im_dim = torch.FloatTensor(dim).repeat(1,2)


            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()


            output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)
            global label_list
            global flag
            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
                picture = orig_im
                label_list = list(map(lambda x: write(x, orig_im), output))
                print("label_list : ", label_list)
                collision(label_list)
                flag = 0

                #이미지 저장하는 코드
                #cv2.imwrite('yolo/static/images/fire_accident.jpg',orig_im)

                ret2,jpeg2 = cv2.imencode('.jpg',orig_im)
                detect_image_byte = jpeg2.tobytes()

                yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + detect_image_byte + b'\r\n\r\n')
                #cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue

            output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim

            im_dim = im_dim.repeat(output.size(0), 1)
            output[:,[1,3]] *= frame.shape[1]
            output[:,[2,4]] *= frame.shape[0]
            #전역변수 선언
            picture = orig_im
            label_list = list(map(lambda x: write(x, orig_im), output))
            print("label_list : ", label_list)
            collision(label_list)
            flag = 0
            #cv2.imshow("frame", orig_im)

            #이미지 저장하는 코드

            ret2,jpeg2 = cv2.imencode('.jpg',orig_im)

            detect_image_byte = jpeg2.tobytes()
            yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + detect_image_byte + b'\r\n\r\n')
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1

            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
        else:
            break

def video_yolo_ready1():
    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "yolov3.weights"
    num_classes = 5
    confidence = 0.6
    nms_thesh = 0.5
    start = 0
    CUDA = torch.cuda.is_available()


    num_classes = 5
    bbox_attrs = 5 + num_classes

    model = Darknet(cfgfile)
    model.load_weights(weightsfile)

    model.net_info["height"] = 416 # 160을 보다 낮은 수 넣으면속도는 빨라짐(단,32의배수값만넣어야함)
    inp_dim = int(model.net_info["height"])

    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    model.eval()

    videofile = 'yolo/static/videos/cctv1_video.mp4'

    cap = cv2.VideoCapture(videofile)
    assert cap.isOpened(), 'Cannot capture source'

    global frames
    global picture
    frames = 0

    start = time.time()
    while cap.isOpened():

        ret, frame = cap.read()
        if ret:
            img, orig_im, dim = prep_image(frame, inp_dim)

            im_dim = torch.FloatTensor(dim).repeat(1,2)


            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            with torch.no_grad():
                output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)
            global label_list
            global flag
            if type(output) == int:
                print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

                picture = orig_im
                label_list = list(map(lambda x: write(x, orig_im), output))
                print("label_list : ", label_list)
                collision(label_list)

                flag = 0
                #이미지 저장하는 코드


                ret2,jpeg2 = cv2.imencode('.jpg',orig_im)
                detect_image_byte = jpeg2.tobytes()
                yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + detect_image_byte + b'\r\n\r\n')
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue




            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)

            output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
            output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2

            output[:,1:5] /= scaling_factor

            for i in range(output.shape[0]):
                output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
                output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])

            classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))

            picture = orig_im
            label_list = list(map(lambda x: write(x, orig_im), output))
            print(label_list)
            collision(label_list)
            flag = 0


            ret2,jpeg2 = cv2.imencode('.jpg',orig_im)
            detect_image_byte = jpeg2.tobytes()
            yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + detect_image_byte + b'\r\n\r\n')
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames+=1
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
        else:
            break;

def video_yolo_ready2():
    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "yolov3.weights"
    num_classes = 5
    confidence = 0.6
    nms_thesh = 0.5
    start = 0
    CUDA = torch.cuda.is_available()


    num_classes = 5
    bbox_attrs = 5 + num_classes

    model = Darknet(cfgfile)
    model.load_weights(weightsfile)

    model.net_info["height"] = 416 # 160을 보다 낮은 수 넣으면속도는 빨라짐(단,32의배수값만넣어야함)
    inp_dim = int(model.net_info["height"])

    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    model.eval()

    videofile = 'yolo/static/videos/cctv3_video.mp4'

    cap = cv2.VideoCapture(videofile)
    assert cap.isOpened(), 'Cannot capture source'

    global frames
    global picture
    frames = 0

    start = time.time()
    while cap.isOpened():

        ret, frame = cap.read()
        if ret:
            img, orig_im, dim = prep_image(frame, inp_dim)

            im_dim = torch.FloatTensor(dim).repeat(1,2)


            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            with torch.no_grad():
                output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)
            global label_list
            global flag
            if type(output) == int:
                print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

                picture = orig_im
                label_list = list(map(lambda x: write(x, orig_im), output))
                print("label_list : ", label_list)
                collision(label_list)

                flag = 0
                #이미지 저장하는 코드


                ret2,jpeg2 = cv2.imencode('.jpg',orig_im)
                detect_image_byte = jpeg2.tobytes()
                yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + detect_image_byte + b'\r\n\r\n')
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue




            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)

            output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
            output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2

            output[:,1:5] /= scaling_factor

            for i in range(output.shape[0]):
                output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
                output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])

            classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))

            picture = orig_im
            label_list = list(map(lambda x: write(x, orig_im), output))
            print(label_list)
            collision(label_list)
            flag = 0


            ret2,jpeg2 = cv2.imencode('.jpg',orig_im)
            detect_image_byte = jpeg2.tobytes()
            yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + detect_image_byte + b'\r\n\r\n')
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames+=1
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
        else:
            break;

def label_list_get(request):
    global label_list
    label_dic = {}
    k=1
    for i in label_list:
        label_dic[k] = i[0]
        k+=1
    if request.is_ajax():
        return HttpResponse(json.dumps(label_dic),'application/json')

@gzip.gzip_page
def stream_yolo_start(request):
    try:
        return StreamingHttpResponse(stream_yolo_ready(),content_type="multipart/x-mixed-replace;boundary=frame")
    except HttpResponseServerError as e:
        print("aborted")

@gzip.gzip_page
def video_yolo_start1(request):
    try:
        return StreamingHttpResponse(video_yolo_ready1(),content_type="multipart/x-mixed-replace;boundary=frame")
    except HttpResponseServerError as e:
        print("aborted")

@gzip.gzip_page
def video_yolo_start2(request):
    try:
        return StreamingHttpResponse(video_yolo_ready2(),content_type="multipart/x-mixed-replace;boundary=frame")
    except HttpResponseServerError as e:
        print("aborted")

def reset_data():
    global label_list
    global cnt_length_queue
    global label_acc_queue
    global label_temp_queue
    global frames
    global flag
    global collision_flag

    label_list.clear()
    while cnt_length_queue.qsize():
        cnt_length_queue.get()

    while label_acc_queue.qsize():
        label_acc_queue.get()

    while label_temp_queue.qsize():
        label_temp_queue.get()

    frames = 0
    flag = 0
    collision_flag = -1


def lf_accumulate(request):
    # 3초 = 75프레임
    # 30 프레임 이상 해당 object가 들어오면 해당 오브젝트로 판단.
    global label_list # 현재 라벨 데이터
    global cnt_length_queue # 프레임 당 라벨의 개수를 담은 큐
    global frames # 현재 프레임수
    global label_acc_queue  # 실제 라벨 데이터 큐
    global label_temp_queue #  임시로 저장할 라벨 큐
    global flag #동기화 시켜주기 위한 변수
    global collision_flag # 충돌 판단하는 변수

    person_cnt =0 # 사람object를 셀 변수
    person_sign=0 # 사람object를 신호주는 변수
    cellphone_cnt = 0
    cellphone_sign = 0
    small_fire1_cnt =0
    small_fire1_sign = 0
    big_fire1_cnt=0
    big_fire1_sign=0
    butane_cnt=0
    butane_sign=0
    multitap_cnt=0
    multitap_sign=0

    test_list = []
    while label_acc_queue.qsize():
        label_name = label_acc_queue.get()
        test_list.append(label_name)
        label_temp_queue.put(label_name)
        #라벨의 개수를 세는 부분 end
    # label_temp에서 label_acc로 복사.
    while label_temp_queue.qsize():
        label_name = label_temp_queue.get()
        label_acc_queue.put(label_name)

    if flag == 0 :
        if len(label_list) == 0 :
            temp =[]
            temp.append('null')
            label_list.append(temp);

        cnt_length_queue.put(len(label_list))
        for i in label_list:
            label_acc_queue.put(i[0])

        if  frames >= 8 :
            cnt = cnt_length_queue.get() #실제 라벨 데이터
            for i in range(0,cnt):
                label_acc_queue.get()  #라벨 데이터 뺀다.
        flag = 1

    #라벨의 개수를 세는 부분 start
    print('================lf_accumulate start================')
    print('frames : ', frames)
    print("label_acc_queue : ",test_list)


    #라벨의 개수를 세는 부분 start
    while label_acc_queue.qsize():
        label_name = label_acc_queue.get()
        if label_name == "person":
            person_cnt += 1
        if label_name == "cell phone":
            cellphone_cnt +=1
        if label_name == "small_fire1":
            small_fire1_cnt +=1
        if label_name == "big_fire1":
            big_fire1_cnt += 1
        if label_name == "butane":
            butane_cnt += 1
        if label_name == "multitap":
            multitap_cnt += 1

        label_temp_queue.put(label_name)
        #라벨의 개수를 세는 부분 end

    # label_temp에서 label_acc로 복사.
    while label_temp_queue.qsize():
        label_name = label_temp_queue.get()
        label_acc_queue.put(label_name)

    ## Object를 카운트한 변수를 가지고 sign 주는 부분
    if person_cnt >= 3 :
        person_sign = 1
    else:
        person_sign = -1

    if cellphone_cnt >= 3 :
        cellphone_sign = 1
    else:
        cellphone_sign = -1

    if small_fire1_cnt >= 1:
        small_fire1_sign = 1
    else:
        small_fire1_sign = -1

    if big_fire1_cnt >= 3:
        big_fire1_sign = 1
    else:
        big_fire1_sign = -1

    if multitap_cnt >= 1:
        multitap_sign = 1
    else:
        multitap_sign = -1

    if butane_cnt >= 1:
        butane_sign = 1
    else:
        butane_sign = -1

    global picture
    if big_fire1_sign == 1 or collision_flag == 1:
        cv2.imwrite('yolo/static/images/fire_accident.jpg',picture)

    result_dic = {"person":person_sign,"cellphone":cellphone_sign, "small_fire1":small_fire1_sign, "big_fire1":big_fire1_sign,"butane":butane_sign, "multitap":multitap_sign,"collision_flag":collision_flag} #1은 person, 1이면 검출ok -1이면 검출x
    print(result_dic)
    print('================lf_accumulate end================')
    if request.is_ajax():
        return HttpResponse(json.dumps(result_dic),'application/json')


def mail(request):
    # 지메일 아이디,비번 입력하기
    email_user = 'vkxmfkdl@gmail.com'      #<ID> 본인 계정 아이디 입력
    email_password = 'kzegjrpnjgwiqcks'      #<PASSWORD> 본인 계정 암호 입력
    email_send = 'qkrrudals689@naver.com'         # <받는곳주소> 수신자 이메일 abc@abc.com 형태로 입력

    # 제목 입력
    subject = '화재가 감지되었습니다. 확인부탁드립니다. -Cold Eyes Project-'

    msg = MIMEMultipart()
    msg['From'] = email_user
    msg['To'] = email_send
    msg['Subject'] = subject

    # 본문 내용 입력
    body = '화재가 감지되었습니다. 확인부탁드립니다. -Cold Eyes Project-'
    msg.attach(MIMEText(body,'plain'))


    ############### ↓ 첨부파일이 없다면 삭제 가능  ↓ ########################
    # 첨부파일 경로/이름 지정하기
    filename='yolo/static/images/fire_accident.jpg'
    attachment  =open(filename,'rb')

    part = MIMEBase('application','octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition',"attachment; filename= "+filename)
    msg.attach(part)
    ############### ↑ 첨부파일이 없다면 삭제 가능  ↑ ########################

    text = msg.as_string()
    server = smtplib.SMTP('smtp.gmail.com',587)
    server.starttls()
    server.login(email_user,email_password)

    server.sendmail(email_user,email_send,text)
    server.quit()


def seoul_fireoffice_weather_info(request):
    context=[]

    fireoffices=fireoffice.objects.all().values('address','number').filter(id=9)
    for office_info in fireoffices:
        context.append([office_info['address'],office_info['number']])

    #a=["부천시 소사구 송내동 부천소방서","032-345-1234"]
    #context.append(a)

    # 날씨연동
    api_key="aa3403904ae3cdee09abb3d46f14ee88"
    lat=37.559
    lng = 126.9987
    forecast=forecastio.load_forecast(api_key, lat, lng)

    #날씨 데이터 넣기
    current_forecast=forecast.hourly()
    #print(current_forecast.summary)
    i=1

    for weather_data in current_forecast.data:
        if(i==2):
            break
        #index 1~49
        #index 1 : present
        #index 2~49 : future expect(hour after) ex) index 2 : 1 hour after windspeed, humidity

        context.append(weather_data.windSpeed)
        context.append(weather_data.humidity)
        i=i+1

    if request.is_ajax():
        return HttpResponse(json.dumps(context), "application/json")


def incheon_fireoffice_weather_info(request):
    print("인천=============================================================================================")
    context=[]

    fireoffices=fireoffice.objects.all().values('address','number').filter(id=271)
    for office_info in fireoffices:
        context.append([office_info['address'],office_info['number']])

    # 날씨연동
    api_key="aa3403904ae3cdee09abb3d46f14ee88"
    lat=37.386508
    lng = 126.657711
    forecast=forecastio.load_forecast(api_key, lat, lng)

    #날씨 데이터 넣기
    current_forecast=forecast.hourly()
    #print(current_forecast.summary)
    i=1


    for weather_data in current_forecast.data:
        if(i==2):
            break
        #index 1~49
        #index 1 : present
        #index 2~49 : future expect(hour after) ex) index 2 : 1 hour after windspeed, humidity

        context.append(weather_data.windSpeed)
        context.append(weather_data.humidity)

        i=i+1

    if request.is_ajax():
        return HttpResponse(json.dumps(context), "application/json")


def gwangju_fireoffice_weather_info(request):
    context=[]

    fireoffices=fireoffice.objects.all().values('address','number').filter(id=278)
    for office_info in fireoffices:
        context.append([office_info['address'],office_info['number']])

    #a=["부천시 소사구 송내동 부천소방서","032-345-1234"]
    #context.append(a)

    # 날씨연동
    api_key="aa3403904ae3cdee09abb3d46f14ee88"
    lat=35.153075
    lng = 126.847151
    forecast=forecastio.load_forecast(api_key, lat, lng)

    #날씨 데이터 넣기
    current_forecast=forecast.hourly()
    #print(current_forecast.summary)
    i=1

    for weather_data in current_forecast.data:
        if(i==2):
            break
        #index 1~49
        #index 1 : present
        #index 2~49 : future expect(hour after) ex) index 2 : 1 hour after windspeed, humidity

        context.append(weather_data.windSpeed)
        context.append(weather_data.humidity)
        i=i+1

    if request.is_ajax():
        return HttpResponse(json.dumps(context), "application/json")

def collision(label_data): #충돌 판단 알고리즘
    global collision_flag
    collision_flag = -1
    for i in range(0,len(label_data)):
        for k in range(i, len(label_data)):
            if label_data[i][0] =="small_fire1" and label_data[k][0]=="butane":
                result_collision = collision_compare(label_data[i],label_data[k])
                if result_collision == 1:
                    print('충돌 발생!!!!!!!!!!!!!!!!!!!!!!!')
                    collision_flag = 1
                    return 1

            if label_data[i][0] =="small_fire1" and label_data[k][0]=="multitab":
                result_collision = collision_compare(label_data[i],label_data[k])
                if result_collision:
                    print('충돌 발생!!!!!!!!!!!!!!!!!!!!!!!')
                    collision_flag = 1
                    return 1
    return -1

def collision_compare(one ,two):
    one_left = one[1]
    one_right = one[3]
    one_top = 416-one[2]
    one_bottom = 416-one[4]

    two_left = two[1]
    two_right = two[3]
    two_top = 416-one[2]
    two_bottom = 416-one[4]

    print("one1111111111111111111111111111111111111111111111111111111111111111111")
    print("one:",one_left,one_right,one_bottom,one_top)
    print("two:",two_left,two_right,two_bottom,two_top)
    if one_left <two_right and one_right > two_left and one_top > two_bottom and one_bottom <two_top:
        return 1
    else:
        return -1
