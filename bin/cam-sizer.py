import cv2

#capture from camera at location 0
cap = cv2.VideoCapture(0)

#set the width and height
cap.set(3,1920)
cap.set(4,1080)


while True:
    ret, img2 = cap.read()
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img2[0:720, 320:1600] # cropped to 1280 x 720
    cv2.imshow("input", img)

    #cv2.imshow("thresholded", imgray*thresh2)
	# operations on the frame come here

    key = cv2.waitKey(10) & 0xEFFFFF
    print "keyboard input:{} img shape: {}".format(key, img.shape)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows() 