#!/usr/bin/python
# image1 1024 x 523
import cv2

image = cv2.imread('image1.jpg')
display_w = 204.0
# calculate ratio of the new image to the old
r = display_w/image.shape[1]
dim = (int(display_w), int(image.shape[0] * r))
resized = cv2.resize(image, dim, interpolation = cv2.INTER_CUBIC)
cv2.imshow('resized', resized)
cv2.waitKey(0)
'''

'''

