# YOLOv8_for_ROS2
ROS2 강의에서 YOLOv8 사물인식 파트에 대한 강의 자료

# Ubuntu 20.04, 22.04 초기 설정 및 YOLOv7 환경 구축

## 처음 우분투 세팅

```bash
sudo apt update
sudo apt upgrade
sudo apt install python3-pip
sudo apt install python3.8-venv
```

## 가상환경 만들고 실행하기

```bash
python3.8 -m venv yolo
source yolo/bin/activate
```

## 패키지 다운로드(Clone)하기

```bash
git clone https://github.com/Nyan-SouthKorea/YOLOv7_with_depthmap.git
```

## 필요 라이브러리 설치하기
1. 편한 방법 : 
```
pip install -r requirments.txt
```

2. 커스텀 설치 : 같이 사용하는 다른 모듈의 종속성에 영향을 미치는 경우(예. ROS2의 경우 numpy 버전이 issue)
아래는 `requirements.txt`의 내용이므로, 버전에 맞게 설치 할 것
```
# Base ----------------------------------------
matplotlib>=3.2.2
numpy>=1.18.5,<1.24.0
opencv-python>=4.1.1
Pillow>=7.1.2
PyYAML>=5.3.1
requests>=2.23.0
scipy>=1.4.1
torch>=1.7.0,!=1.12.0
torchvision>=0.8.1,!=0.13.0
tqdm>=4.41.0
protobuf<4.21.3

# Logging -------------------------------------
tensorboard>=2.4.1
# wandb

# Plotting ------------------------------------
pandas>=1.1.4
seaborn>=0.11.0

# Extras --------------------------------------
ipython  # interactive notebook
psutil  # system utilization
thop  # FLOPs computation
```


## 패키지가 제대로 설치되었는지 확인하기
```bash
source yolo/bin/activate
python
```
여기서 아래 처럼 출력되면 문제 없음(컴퓨터 환경에 따라 GPU를 사용할 수 없는 환경일 수도 있음)
```bash
>>> from YOLOv7_with_depthmap import YOLOv7
>>> model = YOLOv7()
GPU 사용 가능
```

## 강의 자료 코드 로컬에 불러오기

1. 우분투 터미널 실행후 패키지다 다운로드 되길 원하는 경로로 이동한다

    ```bash
    git clone https://github.com/Nyan-SouthKorea/YOLOv7_with_depthmap.git
    ```
    clone된 해당 폴더로 이동

2. VS코드 실행(방금 위에서 생성한 yolo 가상환경 연결하기)

3. Ctrl + Shift + p 버튼을 누르고 "interpreter" 라고 검색하면 아래의 버튼으로 유도됨
![alt text](https://github.com/Nyan-SouthKorea/YOLOv7_with_depthmap/blob/main/README_images/image_1.png)


4. Enter interpreter path 선택
![alt text](https://github.com/Nyan-SouthKorea/YOLOv7_with_depthmap/blob/main/README_images/image_2.png)

5. Find -> .. -> yolo -> bin -> python3 선택

6. VS코드 종료 후 재 실행

7. 아무 코드나 실행해보자. 이 때 select interpreter에서 yolo가 보이면 선택

8. 이후 VS코드 관련 extension을 설치하라는 안내가 나오면 모두 설치해준다. 그리고 실행이 안되면 VS코드 재실행 반복

9. 우리가 만든 yolo라는 가상환경으로 코드를 돌릴 수 있도록 설정 마침

## 테스트 코드(실시간 webcam 인퍼런스)
```python
from YOLOv7_with_depthmap import YOLOv7
import cv2
import numpy as np

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
```
![alt text](https://github.com/Nyan-SouthKorea/YOLOv7_with_depthmap/blob/main/README_images/demo_video_1.gif)


## 테스트 코드(폴더 안에 있는 이미지들 인퍼런스) (더미 depth map도 생성해 봅니다)
```python
from YOLOv7_with_depthmap import YOLOv7
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image
import io

def show_image(image):
    # 이미지를 BGR에서 RGB로 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 이미지를 출력할 수 있는 형태로 변환
    _, buf = cv2.imencode('.png', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    display(Image(data=buf.tobytes()))

model = YOLOv7()
path = 'test images'
for img_name in os.listdir(path):
    # 이미지 읽기
    img = cv2.imread(f'{path}/{img_name}')
    
    # dummy img 생성
    h, w, c = img.shape
    depth_map = np.random.randint(0, 256, (w, h), dtype=np.uint8)

    # 추론
    result = model.detect(bgr_img=img, depth_map=depth_map)
    print(result)
    show_image(model.draw())
```

![alt text](https://github.com/Nyan-SouthKorea/YOLOv7_with_depthmap/blob/main/README_images/demo_video_1.gif)


## 라이센스
YOLOv7 라이센스를 따릅니다. 
https://github.com/WongKinYiu


## 관련 링크
Naver Blog : https://blog.naver.com/112fkdldjs 