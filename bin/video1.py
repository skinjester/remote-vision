import cv2

#capture from camera at location 0
cv2.namedWindow('input', cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)
#set the width and height
cap.set(3,1920)
cap.set(4,1080)

def read():
	return cv2.flip(cv2.transpose(cap.read()[1]),1)

while True:
    img = read()
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("input", img)

    #cv2.imshow("thresholded", imgray*thresh2)
	# operations on the frame come here

    key = cv2.waitKey(10)
    if key == 27:
        break


cv2.destroyAllWindows() 
cv2.VideoCapture(0).release()





