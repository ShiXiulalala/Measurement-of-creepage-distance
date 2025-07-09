import cv2
import sys
from camera_config import camera_config

def getRectifyTransform(width, height, K1, D1, K2, D2, R, T):
    #得出进行立体矫正所需要的映射矩阵 
    # 左校正变换矩阵、右校正变换矩阵、左投影矩阵、右投影矩阵、深度差异映射矩阵
    R_l,R_r,P_l,P_r,Q, roi_left, roi_right = cv2.stereoRectify(K1, D1, K2, D2,
                                            (width, height),R, T,
                                            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
                                        # # 标志CALIB_ZERO_DISPARITY，它用于匹配图像之间的y轴
                                    
    # 计算畸变矫正和立体校正的映射变换。
    map_lx, map_ly = cv2.initUndistortRectifyMap(K1, D1, R_l, P_l, (width,height),cv2.CV_32FC1)
    map_rx, map_ry = cv2.initUndistortRectifyMap(K2, D2, R_r, P_r, (width,height),cv2.CV_32FC1)

    return map_lx, map_ly,map_rx, map_ry, Q

def get_rectify_img(imgL, imgR, map_lx, map_ly, map_rx, map_ry):
    rec_img_L = cv2.remap(imgL,map_lx, map_ly,  cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)  # 使用remap函数完成映射
    rec_img_R = cv2.remap(imgR,map_rx, map_ry,  cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    return rec_img_L, rec_img_R

# 打开摄像头
cap_left = cv2.VideoCapture(2)
cap_right = cv2.VideoCapture(0)


# 获取视频帧的宽度和高度
frame_width = 1920
frame_height = 1080
fps_left = cap_left.get(cv2.CAP_PROP_FPS)
fps_right = cap_right.get(cv2.CAP_PROP_FPS)

cap_left.set(3, frame_width)
cap_left.set(4, frame_height)
cap_right.set(3, frame_width)
cap_right.set(4, frame_height)

# 创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_left = cv2.VideoWriter('Videos/3_27/left.avi', fourcc, fps_left, (frame_width, frame_height))
out_right = cv2.VideoWriter('Videos/3_27/right.avi', fourcc, fps_right, (frame_width, frame_height))

config = camera_config()
K1 = config.K1
D1 = config.D1
K2 = config.K2
D2 = config.D2
R = config.R
T = config.T
Q = config.Q

map_lx, map_ly,map_rx, map_ry, Q = getRectifyTransform(frame_width, frame_height, K1, D1, K2, D2, R, T)

 
if not out_left.isOpened():
    print("Error: Could not open the output video for write")
    sys.exit(1)
if not out_right.isOpened():
    print("Error: Could not open the output video for write")
    sys.exit(1)

while cap_left.isOpened() and cap_right.isOpened():
    ret_left, frame_left = cap_left.read()
    if not ret_left:
        print("Can't receive left frame (stream end?). Exiting ...")
        break
 
    ret_right, frame_right = cap_right.read()
    if not ret_right:
        print("Can't receive right frame (stream end?). Exiting ...")
        break

    # frame_right = cv2.rotate(frame_right, cv2.ROTATE_180)

    frame_left, frame_right = get_rectify_img(frame_left, frame_right, map_lx, map_ly, map_rx, map_ry)

    # 写入帧
    out_left.write(frame_left)
    out_right.write(frame_right)
 
    # 显示帧
    cv2.namedWindow('frame_left', cv2.WINDOW_NORMAL)
    cv2.imshow('frame_left', frame_left)
    cv2.namedWindow('frame_right', cv2.WINDOW_NORMAL)
    cv2.imshow('frame_right', frame_right)
 
    # 按下 'q' 键退出循环
    if cv2.waitKey(int(1000/fps_left)) & 0xFF == ord('q'):
        break
 
cap_left.release()
cap_right.release()
out_left.release()
out_right.release()
cv2.destroyAllWindows()