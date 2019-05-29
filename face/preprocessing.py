import os
import numpy as np
import cv2
import pickle
import tensorflow as tf
from tqdm import tqdm

# 현재파일의 절대경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 경로의 지정폴더에 연결
face_dir = os.path.join(BASE_DIR, 'face-images')

# 매개변수에 있는 얼굴인식 모델을 읽어오기?
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt','res10_300x300_ssd_iter_140000.caffemodel')

count = 0
label_name = {}
y_label = []
x_data = []
dl = []

# face_dir 폴더의 경로, 폴더명, 파일명 담기
for root, dirs, files in os.walk(face_dir):
    face_count = 0
    for file in files:
        # 파일명이 jpg로 끝나는 파일들
        if file.endswith('jpg'):
            # 파일의 경로
            path = os.path.join(root, file)
            # jpg 파일 읽기
            face_img = cv2.imread(path)
            # 사진의 크기 담기
            (h, w) = face_img.shape[:2]
            # 이미지 전처리(사이즈조정, 평균빼기, 이미지 채널바꾸기)
            blob = cv2.dnn.blobFromImage(cv2.resize(face_img, (300, 300)) , 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            # 얼굴모델에 이미지 입력
            net.setInput(blob)
            #
            detections = net.forward()

            # 찾아낸 모델 값의 갯수를 i 에 담기
            # 찾은 모델값이 실제 얼굴인지 조건으로 걸러주기
            for i in range(0, detections.shape[2]):
                if detections[0,0,i,2] > 0.8 and detections[0,0,i,3] > 0 and detections[0,0,i,3] < 1 and \
                        detections[0, 0, i, 4] > 0 and detections[0, 0, i, 4] < 1 and \
                        detections[0, 0, i, 5] > 0 and detections[0, 0, i, 5] < 1 and \
                        detections[0, 0, i, 6] > 0 and detections[0, 0, i, 6] < 1:

                    # 실제 얼굴의 x축의 시작과 끝, y축의 시작과 끝 위치에 사진크기 곱하기
                    box = detections[0,0,i,3:7] * np.array([w,h,w,h])
                    # 값을 타입을 int로 바꾸기
                    (startX, startY, endX, endY) = box.astype('int')
                    # 사진의 얼굴의 사이즈를 조정
                    mod_face = cv2.resize(face_img[startY:endY,startX:endX], dsize=(28,28),
                                          interpolation=cv2.INTER_AREA)
                    # np.array 타입으로 형변환
                    image_array = np.array(mod_face, 'uint8')

                    # x_data 리스트에 담기
                    x_data.append(image_array)

                    # 파일의 상위 위치이름을 수정(공백제거 및 소문자)
                    label = os.path.basename(root).replace(" ","-").lower()

                    # 수정됨 이름이 label_name 에 없을시
                    if label not in label_name:
                        label_name[label] = count
                        count += 1
                        print(count)
                        if label_name[label] not in dl:
                            dl.append(label_name[label])
                    name_ = label_name[label]

                    y_label.append((name_))
            face_count += 1
            if face_count > 30:
                break

train_rate = 0.7
valid_rate = 0.2
curr_index = 0
train_data = []
valid_data = []
test_data = []
train_label = []
valid_label = []
test_label = []


for name in dl:
    label_idx = y_label.count(name)

    train_idx = curr_index + int(label_idx * train_rate)
    valid_idx = train_idx + int(label_idx * valid_rate)
    train_data.extend(x_data[curr_index:train_idx])
    train_label.extend(y_label[curr_index:train_idx])
    valid_data.extend(x_data[train_idx:valid_idx])
    valid_label.extend((y_label[train_idx:valid_idx]))
    test_data.extend(x_data[valid_idx:curr_index+label_idx])
    test_label.extend(y_label[valid_idx:curr_index+label_idx])
    curr_index += label_idx





train_data = np.array(train_data)
train_label = np.array(train_label)
valid_data = np.array(valid_data)
valid_label = np.array(valid_label)
test_data = np.array(test_data)
test_label = np.array(test_label)
pp_data = (train_data, train_label, valid_data, valid_label, test_data, test_label)

print(train_label)
print(valid_label)
print(test_label)

# 나눠진 데이터를 피클 파일로 저장
with open('pp_data.pickle', 'wb') as f:
    pickle.dump(pp_data, f)



# class Data:
#     def __init__(self):
#         self.inputs = []
#         self.labels = []
#
# d = Data()
# print(y_label)
# for labels in dl:
#     print(labels)
#     count = y_label.count(labels)
#     print(curr_index)
#
#     print(y_label[curr_index:curr_index+count])
#     train_index = curr_index + int(count*train_rate)
#     valid_index = train_index + int(count*valid_rate)
#     print(train_index, valid_index)
#     # ******************************************** ******** ********
#     curr_index += count






