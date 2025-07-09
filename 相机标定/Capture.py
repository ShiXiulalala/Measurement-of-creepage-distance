import cv2
 
cap = cv2.VideoCapture(0)  # 打开默认摄像头
 
# 设置分辨率为640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
 
i=0
while True:
    ret, frame = cap.read()  # 读取帧
    if not ret:
        print("无法获取图像")
        break

    if i%10==0: 
        cv2.imwrite('Photos/226/{}.jpg'.format(i),frame)
    i+=1
    cv2.namedWindow('Camera Resolution', cv2.WINDOW_NORMAL)
    cv2.imshow('Camera Resolution', frame)  # 显示图像
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()
