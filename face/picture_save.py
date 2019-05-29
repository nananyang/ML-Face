import cv2


vs = cv2.VideoCapture(0)
i = 0

while True:
    _, frame = vs.read()

    if frame is None:
        break

    picture = cv2.resize(frame, dsize=(300,300))
    cv2.imwrite('face-images/sangjun/{0}.jpg'.format(i),picture)
    i+=1
    if i%128 == 0:
        print(i)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q') or i == 1024: break

vs.release()
cv2.destroyAllWindows()
