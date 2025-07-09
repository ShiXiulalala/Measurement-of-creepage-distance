import cv2
import numpy as np
from scipy.signal import savgol_filter  # 用于平滑边缘点
from camera_config import camera_config
from scipy.optimize import minimize_scalar

config = camera_config()
scale_factor = config.scalefactor
ema_alpha = 0.3  # EMA的权重因子
contourArea_thresh = 0

def ellipse_error(cx, a, b, theta, cy, points):
    """
    计算点集与椭圆的误差
    :param cx: 椭圆中心 x 坐标
    :param a: 半长轴
    :param b: 半短轴
    :param theta: 旋转角度（弧度）
    :param cy: 椭圆中心 y 坐标
    :param points: 点集，形状为 (n, 2)
    :return: 总误差
    """
    # 将点集转换为相对于椭圆中心的坐标
    x_prime = points[:, 0] - cx
    y_prime = points[:, 1] - cy
    
    # 计算椭圆方程值
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    term1 = (x_prime * cos_theta + y_prime * sin_theta) ** 2 / a ** 2
    term2 = (-x_prime * sin_theta + y_prime * cos_theta) ** 2 / b ** 2
    E = term1 + term2 - 1
    
    # 返回总误差
    return np.sum(E ** 2)

# 去除阴影的函数
def remove_shadows(frame):
    # 转换到HSV空间
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 提取亮度通道（V通道）
    v_channel = hsv_frame[:, :, 2]

    # 阈值设置，亮度小于阈值的部分认为是阴影
    _, v_threshold = cv2.threshold(v_channel, 50, 255, cv2.THRESH_BINARY)

    # 创建一个mask，遮盖阴影区域
    shadow_removed_frame = cv2.bitwise_and(frame, frame, mask=v_threshold)

    return shadow_removed_frame


def line_smooth(points):
    sorted_points = points[np.argsort(points[:, 0])]

    smoothed_y = savgol_filter(sorted_points[:,1], 5, 1)  # 窗口大小为5，2阶多项式
    sorted_points[:, 1] = smoothed_y
    return sorted_points

def length_calculate(points, img):
    edge_length = 0
    # 计算平滑后边缘的长度
    for i in range(1, len(points)):
        dist = np.linalg.norm(points[i] - points[i - 1])  # 计算欧几里得距离
        cv2.line(img, points[i], points[i - 1], (255, 255, 255), 4)
        edge_length += dist
    return edge_length


# 处理单帧图像
def process_frame(boxes, classes, scores, seg_imgs, queue_length):
    test = np.zeros_like(seg_imgs[0], dtype=np.uint8)
    mask_img = np.zeros_like(seg_imgs[0], dtype=np.uint8)
    line_img = np.zeros_like(seg_imgs[0], dtype=np.uint8)
    # detection_img = frame.copy()
    test = None
    ema_edge_length = None
    for box, cate, score, seg_img in zip(boxes, classes, scores, seg_imgs):
        # skip if class is not a arrester
        if cate != 0:
            continue
        # skip if conf < 0.9
        seg_img = seg_img *128
        retval, binary_image = cv2.threshold(seg_img, 0, 255, cv2.THRESH_BINARY)
        test = binary_image
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_points = max(contours, key=cv2.contourArea)
        contour_points = contour_points.squeeze()
        # print(np.array(contour_points).shape)
        print(f"原始轮廓点数: {len(contour_points)}")
        if len(contour_points) == 0:
            print("轮廓点数为零")
            continue
        
        LRcutLenth = 0
        # 获取左侧点集合
        left_side_points = contour_points[contour_points[:, 1] < (box[1] + box[3]) // 2]
        # print(len(left_side_points))
        left_side_points = left_side_points[(box[2] - left_side_points[:, 0]) > LRcutLenth]
        Filtered_points_left = left_side_points[(left_side_points[:, 0] - box[0]) > LRcutLenth]

        # print(f"左侧点数: {len(left_side_points)}")
        if len(Filtered_points_left) < 5:
            print("左侧点不足，跳过该目标")
            continue

        sorted_points_left = line_smooth(Filtered_points_left)

        left_edge_length = length_calculate(sorted_points_left, line_img)

        right_side_points = contour_points[contour_points[:, 1] > (box[1] + box[3]) // 2]
        right_side_points = right_side_points[(box[2] - right_side_points[:, 0]) > LRcutLenth]
        Filtered_points_right = right_side_points[(right_side_points[:, 0] - box[0]) > LRcutLenth]

        # print(f"右侧点数: {len(right_side_points)}")
        if len(Filtered_points_right) < 5:
            print("右侧点不足，跳过该目标")
            continue

        sorted_points_right = line_smooth(Filtered_points_right)

        right_edge_length = length_calculate(sorted_points_right, line_img)

        edge_length = (right_edge_length + left_edge_length)/2

        # 在 mask_img 上绘制左侧边缘点
        for point in sorted_points_left:
            cv2.circle(mask_img, tuple(point), 3, (255,255,255), -1)

        for point in sorted_points_right:
            cv2.circle(mask_img, tuple(point), 3, (255,0,255), -1)

        # 转换为实际单位长度
        edge_length_mm = edge_length * scale_factor
        # print(f"边缘长度(毫米): {edge_length_mm}")

        queue_length.put(edge_length_mm)
        if queue_length.full():
            queue_length.get()
        avg_level_mm = np.mean(list(queue_length.queue))
        if ema_edge_length is None:
            ema_edge_length = avg_level_mm
        else:
            ema_edge_length = ema_alpha * avg_level_mm + (1 - ema_alpha) * ema_edge_length
        # # 显示单侧边缘长度
        # cv2.putText(detection_img, f"Edge: {ema_edge_length:.2f} mm", (x1, y1 - 80),
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

        # # 绘制矩形框
        # cv2.rectangle(detection_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.putText(detection_img, f"Conf: {confidence:.2f}", (x1, y1 - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    return mask_img, ema_edge_length-35, line_img, test

def amendment_calculate(boxes, classes, scores, seg_imgs):
    
    test = np.zeros_like(seg_imgs[0], dtype=np.uint8)
    Area_max = 0
    boxM, seg_imgM, contourM,  flag = None, None, None, 0
    for box, cate, score, seg_img in zip(boxes, classes, scores, seg_imgs):
        # skip if class is not a arrester
        if cate != 0:
            continue
        # skip if conf < 0.9
        if score < 0.9:
            continue
        
        retval, binary_image = cv2.threshold(seg_img, 0, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours ,key=cv2.contourArea)
        
        if cv2.contourArea(contour) < contourArea_thresh:
            continue
        elif cv2.contourArea(contour) > Area_max:
            boxM, seg_imgM, contourM , flag= box, seg_img, contour, 1
    
    if flag:
        x1, y1, x2, y2 = boxM
        width = int(x2 - x1)
        height = int(y2 - y1)
        pointsO = []
        pointsI = []
        contour_points = contourM.squeeze()
        for point in contour_points:
            if point[1] - y1 in range(int(0.2*height), int(0.8*height)):
                if point[0] - x1 in range(int(0.2*width), int(0.9*width)):
                    pointsI.append(point)
                else:
                    pointsO.append(point)
            else:
                    pointsO.append(point)

        if len(pointsO) >= 10:
            pointsO = np.array(pointsO)
            ellipse = cv2.fitEllipse(pointsO)
            (center1, axes1, angle1) = ellipse
            # cv2.ellipse(test, ellipse, (255, 255, 255), 3)
            
        # for point in pointsO:
        #     cv2.circle(test, (point[0], point[1]), 3, (255,255,255), -1)  

        axis_1,axis_2 = axes1
        k = axis_1/axis_2

        pointsI = np.array(pointsI)

            #         if len(points) >= 5:
            #             points = points[:, [1, 0]]
                        
            #             # 固定椭圆的参数
        theta = 90.0  # 固定旋转角度（度）
        a = (40/scale_factor)/2       # 固定长轴长度
        cx1 = center1[0]
        cy1 = center1[1]
        b = k*a
        result = minimize_scalar(ellipse_error, args=(a, b, theta, cy1, pointsI), bounds=(0, test.shape[1]), method='bounded')
        cx2 = int(result.x)
        cv2.ellipse(test, (cx2, int(cy1)), (int(a),int(b)), theta, 0, 360, (255, 255, 255), 2)
        print('amendmnt resuilt:')
        print(cx1 - cx2)
        for point in pointsI:
            cv2.circle(test, (point[0], point[1]), 3, (255,255,255), -1)

    return test
