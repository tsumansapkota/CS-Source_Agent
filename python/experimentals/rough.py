import cv2 as cv
import glob
import time
import random

'''load images and shuffle'''
# filenames0 = glob.glob('../saved_res/enemy/*.jpg')
# filenames1 = glob.glob('../saved_res/nothing/*.jpg')
#
# # inputs0 = [(name, 1) for name in filenames0]
# # inputs1 = [(name, 0) for name in filenames1]
# # inputs = inputs0 + inputs1
# inputs = [(name, 1) for name in filenames0] + [(name, 0) for name in filenames1]
# random.shuffle(inputs)
#
# for input in inputs:
#     print(input)

'''image test by fast playing the snapsots'''
# filenames = glob.glob('../saved_res/enemy/*.jpg')
# for image in filenames:
#     print(image)
# for i in range(len(filenames)):
#     image = cv.imread(filenames[i])
#     image = cv.resize(image,(25,50),interpolation=cv.INTER_CUBIC)
#     cv.imshow("MyImage",image)
#     cv.waitKey(2)
# cv.destroyAllWindows()
#
# time.sleep(2)
#
# filenames = glob.glob('../saved_res/nothing/*.jpg')
# for image in filenames:
#     print(image)
# for i in range(len(filenames)):
#     image = cv.imread(filenames[i])
#     image = cv.resize(image,(25,50),interpolation=cv.INTER_CUBIC)
#     cv.imshow("MyImage",image)
#     cv.waitKey(2)
# cv.destroyAllWindows()


# while not cv.waitKey(25) & 0xFF == ord('q'):
#     pass
