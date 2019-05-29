import cv2
import numpy as np
import mlproject

p = mlproject.pp(5,100,10)
p.training()

vs = cv2.VideoCapture('ro.mp4')


net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt','res10_300x300_ssd_iter_140000.caffemodel')

kernel = np.ones((20,20),np.float32) / 400


face_names = ['other','sangjun','taejong','taekjung','yongdae']
count = [0,0,0,0,0]


font=cv2.FONT_HERSHEY_SIMPLEX

while(True):

    _, frame = vs.read()
    if frame is None:
        break
    (h,w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0,detections.shape[2]):
        if detections[0, 0, i, 2] > 0.8 and detections[0, 0, i, 3] > 0 and detections[0, 0, i, 3] < 1 \
            and detections[0, 0, i, 4] > 0 and detections[0, 0, i, 4] < 1 and detections[0, 0, i, 5] > 0 and \
            detections[0, 0, i, 5] < 1 and detections[0, 0, i, 6] > 0 and detections[0, 0, i, 6] < 1:

            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY)= box.astype('int')

            face = frame[startY:endY,startX:endX]
            mod_face = cv2.resize(face,dsize=(28,28),interpolation=cv2.INTER_AREA)
            mod_face = np.array(mod_face)
            #if p.fm.predict(x_test=[mod_face]) != [0]:
            #    frame[startY:endY, startX:endX] = cv2.filter2D(frame[startY:endY, startX:endX], -1, kernel)
            cv2.putText(frame, face_names[p.fm.predict(x_test=[mod_face])[0]], (startX - 5, startY - 5), font, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
            count[p.fm.predict(x_test=[mod_face])[0]] += 1
            #count_1 += 1
            # if p.fm.predict(x_test=[mod_face]) == sangjun:
            #     frame[startY:endY, startX:endX] = cv2.filter2D(frame[startY:endY, startX:endX], -1, kernel)
            #     cv2.putText(frame,'sangjun', (startX - 5, startY - 5), font, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
            #     count_1 += 1
            # elif mlproject.pp.fm.predict(x_test=[mod_face]) == taejong:
            #     frame[startY:endY, startX:endX] = cv2.filter2D(frame[startY:endY, startX:endX], -1, kernel)
            #     cv2.putText(frame, 'taejong', (startX - 5, startY - 5), font, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
            #     count_2 += 1
            # elif mlproject.pp.fm.predict(x_test=[mod_face]) == taekjung:
            #     frame[startY:endY, startX:endX] = cv2.filter2D(frame[startY:endY, startX:endX], -1, kernel)
            #     cv2.putText(frame, 'taekjung', (startX - 5, startY - 5), font, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
            #     count_3 += 1
            # elif mlproject.pp.fm.predict(x_test=[mod_face]) == yongdae:
            #     frame[startY:endY, startX:endX] = cv2.filter2D(frame[startY:endY, startX:endX], -1, kernel)
            #     cv2.putText(frame, 'yongdae', (startX - 5, startY - 5), font, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
            #     count_4 += 1
            # else:
            #     count_0 += 0


            #    print(mlproject.fm.predict(x_test=[mod_face]))



            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)




    cv2.imshow('we',frame)

    #print(count)
    if cv2.waitKey(1) > 0: break


vs.release()
cv2.destroyAllWindows()