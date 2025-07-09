import cv2
import os

i=170
for j in range(1000):
    
    filename = '{}.jpg'.format(i)
    if os.path.exists('Photos/226/'+filename):
        print(filename)
        image = cv2.imread('Photos/226/'+filename)

        img_left = image[0:1080,0:1920]
        img_right = image[0:1080,1920:3840]

        imgl = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        imgr = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        cv2.imshow('1',imgl)
        cv2.waitKey(1)
        cv2.imwrite('data/left/'+filename,imgl)
        cv2.imwrite('data/right/'+filename,imgr)
    i+=10
