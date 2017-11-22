from __future__ import division # so division works like you expect it to
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
# INIT.
# --------

# setup system logging facilities
logging.config.dictConfig(LogSettings.LOGGING_CONFIG)
log = logging.getLogger('logtest-debug')
threadlog = logging.getLogger('logtest-debug-thread')

'''
Camera Manager collects Camera Objects
'''
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
        return self.source[self.current]

    def get(self):
        # returns a pointer to the current camera object
        log.warning('')
        return self.source[self.current]





'''
Camera Object
'''
class WebcamVideoStream(object):

    # the camera has to be provided with a basic landscape width, height
    # because the hardware doesn't capture arbitrary window sizes
    def __init__(self, src, capture_width, capture_height, portrait_alignment, flip_h=False, flip_v=False, gamma=1.0):

        # set camera dimensiions before reading frames
        # requested size is rounded to nearest camera size if non-matching
        self.stream = cv2.VideoCapture(src)
        self.stream.set(3, capture_width)
        self.stream.set(4, capture_height)
        self.width = int(self.stream.get(3))
        self.height = int(self.stream.get(4))
        self.capture_size = [self.width,self.height]

        self.portrait_alignment = portrait_alignment
        self.flip_h = flip_h
        self.flip_v = flip_v
        self.gamma = gamma
        self.stopped = False

        self.table = np.array([((i / 255.0) ** (1.0 / self.gamma)) * 255
        for i in np.arange(0, 256)]).astype("uint8")


        # initial frame to prime the queue
        # the initial capture is aligned on init
        # alignment correction is applied to the capture only
        (self.grabbed, self.frame) = self.stream.read()
        self.frame = self.transpose(self.frame)


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
            self.frame = self.gamma_correct(self.transpose(img))
            threadlog.debug('capture RGB:{}'.format(self.frame))

    def read(self):
        # print "[read] {}".format(self.frame.shape)
        log.info('camera buffer RGB:{}'.format(self.frame.shape))

        # return img_gamma
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
        log.warning('gamma correction enabled')
        return cv2.LUT(img, self.table)

    def stop(self):
        self.stopped = True

class MotionDetector(object):

    def __init__(self, floor, camera, log):
        self.delta_trigger = 0
        self.delta_trigger_history = 0
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
        self.floor = floor
        self.update_hud_log = log
        self.history = []
        self.history_queue_length = 50
        self.forced = False
        self.monitor_msg = None

        # temp
        self._counter_ = 0

        # dataexport
        self.export = open("motiondata-test4.txt","w+")


    def delta_images(self,t0, t1, t2):
        return cv2.absdiff(t2, t0)
    def process(self):
        if self.is_paused:
            self.wasMotionDetected = False
            return

        log.debug('detect motion')
        # history
        self.wasMotionDetected_history = self.wasMotionDetected
        self.delta_count_history = self.delta_count
        self.t_delta_framebuffer = self.delta_images(self.t_minus, self.t_now, self.t_plus)
        retval, self.t_delta_framebuffer = cv2.threshold(self.t_delta_framebuffer, 16, 255, 3)
        cv2.normalize(self.t_delta_framebuffer, self.t_delta_framebuffer, 0, 200, cv2.NORM_MINMAX)
        img_count_view = cv2.cvtColor(self.t_delta_framebuffer, cv2.COLOR_RGB2GRAY)
        self.delta_count = cv2.countNonZero(img_count_view)

        self.delta_trigger = self.add_to_history(self.delta_count*2) + (self.floor*2)

        self.monitor_msg = 'delta_trigger:{} delta_count:{}'.format(self.delta_trigger, self.delta_count,self.delta_count_history)
        log.debug(self.monitor_msg)


        # this ratio represents the number of pixels in motion relative to the total number of pixels on screen
        ratio = self.delta_count
        ratio = float(ratio/(self.camera.width * self.camera.height))
        log.warning('ratio:{:02.3f}'.format(ratio))

        ### GRB export data to textfile
        self.export.write('%d,%d,%d\n'%(self.delta_trigger,self.delta_count,self.delta_count_history))

        if (self.delta_count >= self.delta_trigger and
            self.delta_count_history >= self.delta_trigger):
            # print "[motiondetector] overflow now:{} last:{}".format(self.delta_count,self.delta_count_history)
            self.monitor_msg += ' overflow now:{} last:{}'.format(self.delta_count,self.delta_count_history)
            log.debug(self.monitor_msg)

            self.delta_count = 0 # reseting delta count here, for good reasons but not sure why. Possibly to force the current & previous values to be very different?

        if (self.delta_count >= self.delta_trigger and self.delta_count_history < self.delta_trigger):
            self.delta_count -= int(self.delta_count/2)
            self.update_hud_log('detect','*')
            self.wasMotionDetected = True
            self.monitor_msg += ' movement started'
            log.debug('movement started')

        elif (self.delta_count < self.delta_trigger and self.delta_count_history >= self.delta_trigger):
            self.wasMotionDetected = False
            self.update_hud_log('detect','-')
            log.debug('movement ended')
            self.monitor_msg += ' movement ended'
        else:
            # is this the resting condition?
            self.update_hud_log('detect','-')
            self.wasMotionDetected = False
            log.debug('all motion is beneath threshold:{}'.format(self.floor))


        self._counter_ += 1 # used to index delta_count_history


        lastmsg = '{:0>6}'.format(self.delta_count_history)
        if self.delta_count_history > self.delta_trigger:
            # ratio = 1.0 * self.delta_count_history/self.delta_trigger
            lastmsg = '{:0>6}({:02.3f})'.format(self.delta_count_history,ratio)

        nowmsg = '{:0>6}'.format(self.delta_count)
        if self.delta_count > self.delta_trigger:
            # ratio = 1.0 * self.delta_count/self.delta_trigger
            nowmsg = '{:0>6}({:02.3f})'.format(self.delta_count,ratio)

        self.monitor_msg += str(ratio)

        self.update_hud_log('last',lastmsg)
        self.update_hud_log('now',nowmsg)
        self.refresh_queue()

    def add_to_history(self,value):
        self.history.append(self.delta_count)
        if len(self.history) > self.history_queue_length:
            self.history.pop(0)
        value = int((sum(self.history)*1.2)/(self.history_queue_length)) #GRB history multiplier
        return value

    def force_detection(self):
        self.wasMotionDetected = True

    def isResting(self):
        log.debug('resting...')
        return self.wasMotionDetected == self.wasMotionDetected_history

    def refresh_queue(self):
        log.debug('.')
        self.t_minus = self.t_now
        self.t_now = self.t_plus
        self.t_plus = self.camera.read()
        self.t_plus = cv2.blur(self.t_plus,(20,20))