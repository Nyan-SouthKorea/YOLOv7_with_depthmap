from hubconf import custom
import time
import cv2
import math
import torch
import torchvision
import time
import os
import requests

class Custom_YOLOv7:
    '''
    240524
    A module has been created based on the official YOLOv7 repository to produce inference 
    results in the form of a list of dictionaries when a numpy image is input
    '''
    def __init__(self, model_path, conf_thresh=0.25, nms_thresh=0.45, filter = None):
        '''
        model_path: Path to the YOLOv7 weight file
        center_point: [x, y] The center point of the image for measuring the distance of an object (defaults to the bottom center if not provided)
        roi_box: [x1, y1, x2, y2] Set the region of interest for object recognition (views the entire area if not provided)
        conf_thresh: Set the confidence threshold for object recognition
        nms_thresh: Set the non-maximum suppression threshold for object recognition
        filter: if filter is not None, return classes only in filter. name of the object detection class names should be put like 'person', 'bottle', etc..
        '''
        self.base_weights_check()
        self.model = custom(path_or_model = model_path, conf_thresh=conf_thresh, nms_thresh=nms_thresh)
        self.filter = filter

    def detect(self, bgr_img):
        '''
        return dic_list from image after inference
        img: bgr image from cv2 library
        '''
        # image bgr -> rgb
        self.bgr_img = bgr_img # use this val when drawing
        self.img = cv2.cvtColor(self.bgr_img, cv2.COLOR_BGR2RGB)
        # inference
        start = time.time()
        results = self.model(self.img).pandas().xyxy[0]
        spent_time = round(time.time() - start, 3)
        # post processing
        self.dic_list = []
        for idx, row in results.iterrows():
            bbox = [int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])]
            conf = round(row['confidence'], 3)
            class_no = row['class']
            name = row['name']
            # apply filter
            if self.filter != None:
                if not name in self.filter:
                    continue
            self.dic_list.append({'bbox':bbox, 'conf':conf, 'class_no':class_no, 'name':name, 'inf_time':spent_time})
        return self.dic_list
    
    def draw(self):
        '''
        draw result to self.img by self.dic_list
        '''
        for dic in self.dic_list:
            cv2.rectangle(self.bgr_img, (dic['bbox'][0], dic['bbox'][1]), (dic['bbox'][2], dic['bbox'][3]), (0,0,255), 2)
            text = f'{dic["name"]}:{dic["conf"]}'
            cv2.putText(self.bgr_img, text, (dic['bbox'][0], dic['bbox'][1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        try: fps = int(1/dic['inf_time'])
        except: fps = 99
        cv2.putText(self.bgr_img, f'fps: {fps}', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        return self.bgr_img
    
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
    model = Custom_YOLOv7(model_path = 'weights/yolov7.pt')
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if ret == False:
            print('웹캠 수신 실패. 프로그램 종료')
            break
        result = model.detect(bgr_img = img)
        print(result)
        cv2.imshow('YOLOv7 test', model.draw())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    