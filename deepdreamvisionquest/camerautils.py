import numpy as np
import cv2
import datetime
from threading import Thread
import sys
sys.path.append('../bin') #  point to directory containing LogSettings
import logging
import logging.config
import LogSettings # global log settings template

# --------
# INIT
# --------

# setup system logging facilities
logging.config.dictConfig(LogSettings.LOGGING_CONFIG)
log = logging.getLogger('logtest-debug')
threadlog = logging.getLogger('logtest-debug-thread')


class WebcamVideoStream(object):

    # the camera has to be provided with a basic landscape width, height
    # because the hardware doesn't capture arbitrary window sizes
    def __init__(self, src, capture_width, capture_height, portrait_alignment, gamma=1.0):

        # set camera dimensiions before reading frames
        # requested size is rounded to nearest camera size if non-matching
        self.stream = cv2.VideoCapture(src)
        self.stream.set(3, capture_width)
        self.stream.set(4, capture_height)
        self.width = int(self.stream.get(3))
        self.height = int(self.stream.get(4))
        self.capture_size = [self.width,self.height]

        self.portrait_alignment = portrait_alignment
        self.gamma = gamma

        # initial frame to prime the queue
        # the initial capture is aligned on init
        # because any alignment correction is applied to the capture only
        (self.grabbed, self.frame) = self.stream.read()
        if self.portrait_alignment:
            # self.frame = cv2.flip(cv2.transpose(self.frame),1)
            self.frame = cv2.transpose(self.frame)

        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        threadlog.debug('started new thread')
        return self
    def update(self):
        # loop until the thread is stopped
        while True:
            if self.stopped:
                return

            _,img = self.stream.read()
            if self.portrait_alignment:
                #img = cv2.flip(cv2.transpose(img),1)
                img = cv2.transpose(img)


            threadlog.debug('capture RGB:{}'.format(img.shape))
            self.frame = img

    def read(self):
        # print "[read] {}".format(self.frame.shape)
        log.info('camera buffer RGB:{}'.format(self.frame.shape))
        # invGamma = 1.0 / self.gamma
        # table = np.array([((i / 255.0) ** invGamma) * 255
        #     for i in np.arange(0, 256)]).astype("uint8")

        # # apply gamma correction using the lookup table
        # img_gamma =  cv2.LUT(self.frame, table)

        # return img_gamma
        return self.frame

    def realign(self):
        self.portrait_alignment = not self.portrait_alignment

    def stop(self):
        self.stopped = True

class MotionDetector(object):

    def __init__(self, delta_trigger, camera, log):
        self.delta_trigger = delta_trigger
        self.delta_trigger_history = delta_trigger
        self.delta_count = 0
        self.delta_count_history = 0
        self.camera = camera

        self.t_minus = self.camera.read()
        self.t_now = self.camera.read()
        self.t_plus =self.camera.read()

        self.width = self.t_minus.shape[1]
        self.height = self.t_minus.shape[0]

        self.t_delta_framebuffer = np.zeros((self.height, self.width ,3), np.uint8) # empty img
        self.wasMotionDetected = False
        self.wasMotionDetected_history = False
        self.is_paused = False
        self.floor = 30000 # changed to lover value at night
        self.update_hud_log = log
        self.history = []
        self.history_queue_length = 50

    def delta_images(self,t0, t1, t2):
        return cv2.absdiff(t2, t0)
    def process(self):
        if self.is_paused:
            self.wasMotionDetected = False
            return

        log.info('detect motion')
        # history
        self.wasMotionDetected_history = self.wasMotionDetected
        self.delta_count_history = self.delta_count
        self.t_delta_framebuffer = self.delta_images(self.t_minus, self.t_now, self.t_plus)
        retval, self.t_delta_framebuffer = cv2.threshold(self.t_delta_framebuffer, 16, 255, 3)
        cv2.normalize(self.t_delta_framebuffer, self.t_delta_framebuffer, 0, 255, cv2.NORM_MINMAX)
        img_count_view = cv2.cvtColor(self.t_delta_framebuffer, cv2.COLOR_RGB2GRAY)
        self.delta_count = cv2.countNonZero(img_count_view)

        self.delta_trigger = self.add_to_history(self.delta_count) + self.floor
        #print 'avg:raw {}:{}'.format(self.delta_trigger, self.delta_count)
        log.info('delta_trigger:{} delta_count:{} delta_history:{}'.format(self.delta_trigger, self.delta_count,self.delta_count_history))

        if (self.delta_count >= self.delta_trigger and
            self.delta_count_history >= self.delta_trigger):
            # print "[motiondetector] overflow now:{} last:{}".format(self.delta_count,self.delta_count_history)
            log.info('detection overflow now:{} last:{}'.format(self.delta_count,self.delta_count_history))
            self.delta_count = 0

        if (self.delta_count >= self.delta_trigger and self.delta_count_history < self.delta_trigger):
            self.delta_count -= int(self.delta_count/2)
            self.update_hud_log('detect','*')
            self.wasMotionDetected = True
            log.info('movement started')

        elif (self.delta_count < self.delta_trigger and self.delta_count_history >= self.delta_trigger):
            self.wasMotionDetected = False
            self.update_hud_log('detect','-')
            log.info('movement ended')
        else:
            self.update_hud_log('detect','-')
            self.wasMotionDetected = False
            #print "[motiondetector] beneath threshold."

        # logging
        lastmsg = '{:0>6}'.format(self.delta_count_history)
        if self.delta_count_history > self.delta_trigger:
            ratio = 1.0 * self.delta_count_history/self.delta_trigger
            lastmsg = '{:0>6}({:02.3f})'.format(self.delta_count_history,ratio)

        nowmsg = '{:0>6}'.format(self.delta_count)
        if self.delta_count > self.delta_trigger:
            ratio = 1.0 * self.delta_count/self.delta_trigger
            nowmsg = '{:0>6}({:02.3f})'.format(self.delta_count,ratio)

        self.update_hud_log('last',lastmsg)
        self.update_hud_log('now',nowmsg)
        self.refresh_queue()

    def add_to_history(self,value):
        self.history.append(self.delta_count)
        if len(self.history) > self.history_queue_length:
            self.history.pop(0)
        value = int(sum(self.history)/(self.history_queue_length))
        # if value < self.floor:
        #     value += self.floor
        return value


    def isResting(self):
        return self.wasMotionDetected == self.wasMotionDetected_history

    def refresh_queue(self):
        #print "---- [motiondetector] refresh queue"
        self.t_minus = self.t_now
        self.t_now = self.t_plus
        self.t_plus = self.camera.read()
        self.t_plus = cv2.blur(self.t_plus,(20,20))