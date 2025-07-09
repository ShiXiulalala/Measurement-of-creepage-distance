import cv2
import time
import sys
import queue
from rknnpool import rknnPoolExecutor
# 图像处理函数，实际应用过程中需要自行修改
from func import myFunc, queue_define
from camera_config import camera_config
import numpy as np

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

def img_show(str, img):
    cv2.namedWindow(str, cv2.WINDOW_NORMAL)
    cv2.imshow(str, img)

def YOLOinit(camera_thread):
    config = camera_config()
    K1 = config.K1
    D1 = config.D1
    K2 = config.K2
    D2 = config.D2
    R = config.R
    T = config.T
    Q = config.Q

    width = 1920
    height = 1080

    queue_lengthL = queue.Queue(maxsize=30)
    queue_lengthR = queue.Queue(maxsize=30)
    queue_define(queue_lengthL, queue_lengthR)

    map_lx, map_ly,map_rx, map_ry, Q = getRectifyTransform(width, height, K1, D1, K2, D2, R, T)

    modelPath = "./yolov8_seg.rknn"
    modelPath_1 = "./527.rknn"
    # 线程数, 增大可提高帧率
    TPEs = 3
    # 初始化rknn池
    pool = rknnPoolExecutor(
        rknnModel=modelPath,
        rknnModel_1=modelPath_1,
        TPEs=TPEs,
        func=myFunc)

    

    if (camera_thread[21].isOpened() and camera_thread[23].isOpened()):
        cap_left = camera_thread[21]
        cap_right = camera_thread[23]
        for i in range(TPEs + 1):
            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()
            frame_left_copy = frame_left.copy()
            frame_right_copy = frame_right.copy()
            frame_left_copy, frame_right = get_rectify_img(frame_left_copy, frame_right_copy, map_lx, map_ly, map_rx, map_ry)
            pool.put(frame_left_copy, frame_right_copy)
        
    frames, loopTime, initTime = 0, time.time(), time.time()

    return frames, cap_left, cap_right, map_lx, map_ly, map_rx, map_ry, pool

def YOLO_run(frames, cap_left, cap_right, map_lx, map_ly, map_rx, map_ry, pool):
    frames += 1
    initTime = time.time()
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    readTime = time.time()
    # print("读取帧时间:\t", readTime - initTime, "秒")
    
    frame_left, frame_right = get_rectify_img(frame_left, frame_right, map_lx, map_ly, map_rx, map_ry)
    #print(frame.shape)
    pool.put(frame_left, frame_right)
    putTime = time.time()
    # print("放入帧时间:\t", putTime - readTime, "秒")
    result, flag = pool.get()
    getTime = time.time()
    # print("获取帧时间:\t", getTime - putTime, "秒")
    #print(frame.shape)
    frame_left, frame_right, mask_imgL, mask_imgR, line_imgL, line_imgR, ema_edge_length, seg_img, test = result[:]
    # print('长度为：'+str(ema_edge_length))
    # if mask_imgL is None:
    #     continue
    # img_show('left_img', frame_left)
    # img_show('right_img', frame_right)
    # img_show('test', seg_img)
    # img_show('mask_imgL', mask_imgL)
    # img_show('mask_imgR', mask_imgR)
    # img_show('line_imgL', line_imgL)
        # img_show('line_imgR', line_imgR)
        # img_show('test', test)

        # print("绝缘子长度：" + ema_edge_length)
    showTime = time.time()
    # print("显示帧时间:\t", showTime - getTime, "秒")

    if frames % 30 == 0:
        print("30帧平均帧率:\t", 30 / (time.time() - loopTime), "帧")
        loopTime = time.time()
    print("总平均帧率\t", frames / (time.time() - initTime))
    return ema_edge_length, mask_imgL

if __name__ == "__main__":
    config = camera_config()
    K1 = config.K1
    D1 = config.D1
    K2 = config.K2
    D2 = config.D2
    R = config.R
    T = config.T
    Q = config.Q

    width = 1920
    height = 1080

    queue_lengthL = queue.Queue(maxsize=30)
    queue_lengthR = queue.Queue(maxsize=30)
    queue_define(queue_lengthL, queue_lengthR)

    cap_left = cv2.VideoCapture(21)
    cap_right = cv2.VideoCapture(23)
    cap_left.set(3, width)
    cap_left.set(4, height)
    cap_right.set(3, width)
    cap_right.set(4, height)
    if not cap_left.isOpened():
        sys.exit("Error: Unable to open left video file/camera.")
    if not cap_right.isOpened():
        sys.exit("Error: Unable to open right video file/camera.")

    map_lx, map_ly,map_rx, map_ry, Q = getRectifyTransform(width, height, K1, D1, K2, D2, R, T)

    modelPath = "./yolov8_seg.rknn"
    modelPath_1 = "./527.rknn"
    # 线程数, 增大可提高帧率
    TPEs = 3
    # 初始化rknn池
    pool = rknnPoolExecutor(
        rknnModel=modelPath,
        rknnModel_1=modelPath_1,
        TPEs=TPEs,
        func=myFunc)


    # 初始化异步所需要的帧
    if (cap_left.isOpened() and cap_right.isOpened()):
        for i in range(TPEs + 1):
            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()
            if not ret_left or not ret_right:
                cap_left.release()
                cap_right.release()
                del pool
                exit(-1)
            frame_left, frame_right = get_rectify_img(frame_left, frame_right, map_lx, map_ly, map_rx, map_ry)
            pool.put(frame_left, frame_right)

    frames, loopTime, initTime = 0, time.time(), time.time()
    while (cap_left.isOpened() and cap_right.isOpened()):
        frames += 1
        initTime = time.time()
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        readTime = time.time()
        # print("读取帧时间:\t", readTime - initTime, "秒")

        if not ret_left or not ret_right:
            break
        frame_left, frame_right = get_rectify_img(frame_left, frame_right, map_lx, map_ly, map_rx, map_ry)
        #print(frame.shape)
        pool.put(frame_left, frame_right)
        putTime = time.time()
        # print("放入帧时间:\t", putTime - readTime, "秒")
        result, flag = pool.get()
        getTime = time.time()
        # print("获取帧时间:\t", getTime - putTime, "秒")
        if flag == False:
            break
        #print(frame.shape)
        frame_left, frame_right, mask_imgL, mask_imgR, line_imgL, line_imgR, ema_edge_length, seg_img, test = result[:]
        # print('长度为：'+str(ema_edge_length))
        # if mask_imgL is None:
        #     continue
        # img_show('left_img', frame_left)
        # img_show('right_img', frame_right)
        # img_show('test', seg_img)
        # img_show('mask_imgL', mask_imgL)
        # img_show('mask_imgR', mask_imgR)
        # img_show('line_imgL', line_imgL)
        # img_show('line_imgR', line_imgR)
        # img_show('test', test)

        # print("绝缘子长度：" + ema_edge_length)

        showTime = time.time()
        # print("显示帧时间:\t", showTime - getTime, "秒")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frames % 30 == 0:
            print("30帧平均帧率:\t", 30 / (time.time() - loopTime), "帧")
            loopTime = time.time()

    print("总平均帧率\t", frames / (time.time() - initTime))
    # 释放cap和rknn线程池
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
    pool.release()
