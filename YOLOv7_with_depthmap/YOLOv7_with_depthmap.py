# 기본
import os
import time
import math

# pypi 설치
import cv2
import numpy as np
import torch
import torchvision
import requests

# 외부 폴더
from hubconf import custom


class YOLOv7:
    '''
    yolov7에 이미지와 depth_map이 투입되면 인식된 사물의 결과와, bbox 내의 depth를 계산하여 함께 반환
    '''
    def __init__(self, model_path='weights/yolov7-tiny.pt', conf_thresh=0.25, nms_thresh=0.45, filter = None):
        '''
        model_path: Path to the YOLOv7 weight file
        center_point: [x, y] The center point of the image for measuring the distance of an object (defaults to the bottom center if not provided)
        roi_box: [x1, y1, x2, y2] Set the region of interest for object recognition (views the entire area if not provided)
        conf_thresh: Set the confidence threshold for object recognition
        nms_thresh: Set the non-maximum suppression threshold for object recognition
        filter: if filter is not None, return classes only in filter. name of the object detection class names should be put like 'person', 'bottle', etc..
        '''
        if torch.cuda.is_available():
            print('GPU 사용 가능')
        else:
            print('GPU 사용 불가. CPU를 사용하기 때문에 추론 속도가 느릴 수 있습니다')
        time.sleep(1)
        self.base_weights_check()
        self.model = custom(path_or_model = model_path, conf_thresh=conf_thresh, nms_thresh=nms_thresh)
        self.filter = filter

    def detect(self, bgr_img, depth_map=None):
        '''
        return dic_list from image after inference
        img: bgr image from cv2 library
        '''
        # image bgr -> rgb
        self.bgr_img = bgr_img # use this val when drawing
        self.img = cv2.cvtColor(self.bgr_img, cv2.COLOR_BGR2RGB)
        # 추론
        start = time.time()
        results = self.model(self.img).pandas().xyxy[0]
        spent_time = round(time.time() - start, 3)
        # 전처리
        self.dic_list = []
        for idx, row in results.iterrows():
            bbox = [int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])]
            conf = round(row['confidence'], 3)
            class_no = row['class']
            name = row['name']
            # filter 적용
            if self.filter != None:
                if not name in self.filter:
                    continue
            # depth 계산
            if depth_map is not None:
                depth = self._get_depth(depth_map, bbox) # depth 추출
            else:
                depth = -1 # depth_map을 넣지 않을 경우 -1 반환

            # self.dic_list에 append
            self.dic_list.append({'bbox':bbox, 'conf':conf, 'class_no':class_no, 'name':name, 'inf_time':spent_time, 'depth':depth})
        return self.dic_list
    
    def draw(self):
        '''
        draw result to self.img by self.dic_list
        '''
        for dic in self.dic_list:
            cv2.rectangle(self.bgr_img, (dic['bbox'][0], dic['bbox'][1]), (dic['bbox'][2], dic['bbox'][3]), (0,0,255), 2)
            text = f'{dic["name"]}:{dic["conf"]}, depth:{dic["depth"]}'
            cv2.putText(self.bgr_img, text, (dic['bbox'][0], dic['bbox'][1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        try: fps = int(1/dic['inf_time'])
        except: fps = 99
        cv2.putText(self.bgr_img, f'fps: {fps}', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        return self.bgr_img
    
    def _get_depth(self, depth_map, bbox, rate=0.3):
        '''
        depth_map에서 bbox영역의 중앙 영역 depth를 반환

        depth_map : 내부 변수
        bbox : 내부 변수
        rate : bbox 면적에서 해당 rate만큼의 중앙 영역의 평균 값을 계산함
        '''
        # rate 반영한 bbox 계산
        x1, y1, x2, y2 = bbox
        # bbox의 가로 세로 길이
        x_len = x2-x1
        y_len = y2-y1
        # rate 반영한 bbox의 가로 세로 길이
        new_x_len = x_len*rate
        new_y_len = y_len*rate
        # rate 반영된 bbox 수치 계산
        new_x1 = x1 + ((x_len-new_x_len)/2)
        new_y1 = y1 + ((y_len-new_y_len)/2)
        new_x2 = x2 - ((x_len-new_x_len)/2)
        new_y2 = y2 - ((y_len-new_y_len)/2)
        new_x1, new_y1, new_x2, new_y2 = int(new_x1), int(new_y1), int(new_x2), int(new_y2)
        # depth_map에서 원하는 부분 crop하여 평균 depth 구하기
        crop_depth = depth_map[new_y1:new_y2, new_x1:new_x2]
        mean_depth = np.mean(crop_depth)
        return round(mean_depth, 2)
    
    def base_weights_check(self):
        url_dic = {'yolov7-tiny.pt':'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt', 
                   'yolov7.pt': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt', 
                   'yolov7-X.pt': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt', 
                   'yolov7-W6.pt': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt'}
        # weights 폴더 생성
        if not os.path.exists('./weights'):
            os.makedirs('./weights')
        # 기본 weight파일들 다운로드 받아주기(모두 다운받는 이유는 사용자가 보고 골라 사용할 수 있게끔)
        for model_name, url in url_dic.items():
            if not os.path.exists(f'./weights/{model_name}'):
                # file download
                response = requests.get(url)
                # response check
                if response.status_code == 200:
                    with open(f'./weights/{model_name}', 'wb') as file:
                        file.write(response.content)
                    print(f'{model_name} downloaded done')
                else:
                    print(f'{model_name} is not downloaded. visit YOLOv7 repository and download by your self')
            else:
                print(f'{model_name} checked')

if __name__ == '__main__':
    model = YOLOv7()
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if ret == False:
            print('웹캠 수신 실패. 프로그램 종료')
            break
        # dummy 데이터 생성
        h, w, c = img.shape
        depth_map = np.random.randint(0, 256, (w, h), dtype=np.uint8)
        # 추론
        result = model.detect(bgr_img=img, depth_map=depth_map)
        print(result)
        cv2.imshow('YOLOv7 test', model.draw())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break