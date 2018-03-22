import cv2

#capture from camera at location 0
cameraID = 1
cap = cv2.VideoCapture(cameraID)

#set the width and height
cap.set(3,3840)
cap.set(4,2160)


while True:
    ret, img2 = cap.read()
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img2
    # img = img2[0:720, 320:1600] # cropped to 1280 x 720
    cv2.imshow("input", img)

    #cv2.imshow("thresholded", imgray*thresh2)
	# operations on the frame come here

    key = cv2.waitKey(10) & 0xEFFFFF
    print "camera: {} keyboard input: {} img shape: {}".format(cameraID, key, img.shape)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows() 