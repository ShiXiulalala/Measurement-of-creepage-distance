import cv2
import os

# 打开视频文件
video1 = cv2.VideoCapture('Videos/3_27/left.avi')
video2 = cv2.VideoCapture('Videos/3_27/right.avi')

# 定义帧计数器
frame_count = 0

# 定义保存帧的文件夹路径
output_folder1 = 'Frames/left'
output_folder2 = 'Frames/right'
if not os.path.exists(output_folder1):
    os.makedirs(output_folder1)
if not os.path.exists(output_folder2):
    os.makedirs(output_folder2)
# 逐帧读取视频
while True:
    # 读取下一帧
    ret1, frame1 = video1.read()
    ret2, frame2 = video2.read()

    # 如果读取失败（例如，达到视频末尾），则跳出循环
    if not ret1:
        break
    if not ret2:
        break
    # 检查当前帧是否是5的倍数
    if frame_count % 5 == 0:
        # 将当前帧保存到指定文件夹，并使用递增的数字进行命名
        cv2.imwrite(os.path.join(output_folder1, 'frame_l{}.jpg'.format(frame_count // 5)), frame1)
        cv2.imwrite(os.path.join(output_folder2, 'frame_r{}.jpg'.format(frame_count // 5)), frame2)
    frame_count += 1

print(frame_count)
# 释放视频对象并关闭所有窗口
video1.release()
video2.release()
cv2.destroyAllWindows()

