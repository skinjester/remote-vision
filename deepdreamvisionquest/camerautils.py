from __future__ import division # so division works like you expect it to
import numpy as np
import cv2
import time
import datetime
from threading import Thread
import sys
sys.path.append('../bin') #  point to directory containing LogSettings
import logging
import logging.config
import LogSettings # global log settings template





# Camera collection
class Cameras(object):
    def __init__(self, source=[], current=0):
        self.source = source
        self.current = current

    def next(self):
        # returns a pointer to the next available camera object
        if (self.current + 1 >= len(self.source)):
            self.current = 0
        return self.source[self.current]

    def previous(self):
        # returns a pointer to the previous available camera object
        if (self.current - 1 < 0):
            self.current = len(self.source) - 1
        return self.source[self.current]

    def set(self,camera_index):
        # returns a pointer to the specified camera object
        self.current = camera_index
        log.critical('cameraID: {}'.format(self.current))
        return self.source[self.current]

    def get(self):
        # returns a pointer to the current camera object
        log.critical('cameraID: {}'.format(self.current))
        return self.source[self.current]


# camera object
class WebcamVideoStream(object):

    # the camera has to be provided with a basic landscape width, height
    # because the hardware doesn't capture arbitrary window sizes
    def __init__(self, src, capture_width, capture_height, portrait_alignment, flip_h=False, flip_v=False, gamma=1.0):

        # set camera dimensions before reading frames
        # requested size is rounded to nearest camera size if non-matching
        self.stream = cv2.VideoCapture(src)
        self.stream.set(3, capture_width)
        self.stream.set(4, capture_height)
        self.width = int(self.stream.get(3))
        self.height = int(self.stream.get(4))
        self.capture_size = [self.width,self.height]

        # image transform and gamma
        self.portrait_alignment = portrait_alignment
        self.flip_h = flip_h
        self.flip_v = flip_v
        self.gamma = gamma
        self.stopped = False

        # generates internal table for gamma correction
        self.table = np.array([((i / 255.0) ** (1.0 / self.gamma)) * 255
        for i in np.arange(0, 256)]).astype("uint8")

        # motion detection
        self.delta_count = 0 # difference between current frame and previous
        self.img_difference = np.zeros((self.height, self.width ,3), np.uint8) # framebuffer for image differencing operations

        # frame buffer housekeeping
        self.rawframe = np.zeros((self.height, self.width ,3), np.uint8) # empty img for initial diff
        (self.grabbed, self.frame) = self.stream.read() # initial frame to prime the queue
        self.frame = self.transpose(self.frame) # alignment correction


    def start(self):
        Thread(target=self.update, args=()).start()
        threadlog.critical('started camera thread')
        return self

    def update(self):
        # loop until the thread is stopped
        while True:
            if self.stopped:
                return

            _, img = self.stream.read()

            # self.rawframe contains non-transformed/non gamma-corrected camera img
            # self.frame contains transformed/gamma-corrected image

            # motion detection
            self.img_difference = cv2.subtract(img, self.rawframe)
            self.img_difference = cv2.cvtColor(self.img_difference, cv2.COLOR_RGB2GRAY)
            _, self.img_difference = cv2.threshold(self.img_difference, 32, 255, cv2.THRESH_TOZERO)
            self.delta_count = cv2.countNonZero(self.img_difference)
            MotionDetector.process()

            # update internal buffers
            self.rawframe = img # unprocessed camera img
            self.frame = self.gamma_correct(self.transpose(img)) # processed camera img

            # logging
            cv2.imshow('diff', _diff)
            threadlog.critical('_countdiff:{}'.format(_count))
            threadlog.debug('camera buffer RGB:{}'.format(self.frame.shape))

    def read(self):
        log.debug('camera buffer RGB:{}'.format(self.frame.shape))
        return self.frame

    def transpose(self, img):
        if self.portrait_alignment: 
            img = cv2.transpose(img)
        if self.flip_v:
            img = cv2.flip(img, 0)
        if self.flip_h:
            img = cv2.flip(img, 1)
        return img

    def gamma_correct(self, img):
        # # apply gamma correction using the lookup table defined on init
        if self.gamma == 1.0:
            return img
        return cv2.LUT(img, self.table)

    def stop(self):
        self.stopped = True

class MotionDetector(object):

    def __init__(self, floor):
        self.delta_trigger = 0
        self.delta_trigger_history = 0
        self.delta_count = 0
        self.delta_count_history = 0
        self.wasMotionDetected = False
        self.wasMotionDetected_history = False
        self.is_paused = False
        self.floor = floor
        self.update_hud_log = log
        self.history = []
        self.history_queue_length = 50
        self.forced = False
        self.monitor_msg = None

        # dataexport
        self.export = open("motiondata/motiondata-test-6.txt","w+")

        # temp (?)
        self._counter_ = 0
        self.elapsed = 0
        self.now = time.time()
        self.counted = 0

        # temp for thread debug
        self.thread_msg = 'none'
        self.stopped = False

    def process(self):
        pass

    def add_to_history(self,value):
        self.history.append(self.delta_count)
        if len(self.history) > self.history_queue_length:
            self.history.pop(0)
        value = int((sum(self.history)*1.2)/(self.history_queue_length)) #GRB history multiplier
        return value

    def force_detection(self):
        pass
        #self.wasMotionDetected = True

    def isResting(self):
        log.debug('resting...')
        return self.wasMotionDetected == self.wasMotionDetected_history

    def refresh_queue(self):
        log.debug('.')
        self.t_minus = self.t_now
        self.t_now = self.t_plus
        self.t_plus = self.camera.read()
        #self.t_plus = cv2.blur(self.t_plus,(20,20))


# --------
# INIT.
# --------

# setup system logging facilities
logging.config.dictConfig(LogSettings.LOGGING_CONFIG)
log = logging.getLogger('logtest-debug')
log.setLevel(logging.CRITICAL)
threadlog = logging.getLogger('logtest-debug-thread')
threadlog.setLevel(logging.CRITICAL)

'''
Camera Manager collects any Camera Objects
'''

# log.debug('*debug message!')
# log.info('*info message!')
# log.error('*error message')
# log.warning('warning message')
# log.critical('critical message')


# floor value compensates for the amount of light in the area
MotionDetector = MotionDetector(floor=0)