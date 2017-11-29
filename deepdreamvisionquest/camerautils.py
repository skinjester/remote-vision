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
        log.debug('cameraID: {}'.format(self.current))
        return self.source[self.current]

    def get(self):
        # returns a pointer to the current camera object
        log.debug('cameraID: {}'.format(self.current))
        return self.source[self.current]


# camera object
class WebcamVideoStream(object):

    # the camera has to be provided with a basic landscape width, height
    # because the hardware doesn't capture arbitrary window sizes
    def __init__(self, src, capture_width, capture_height, portrait_alignment, log, flip_h=False, flip_v=False, gamma=1.0):

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
        self.motiondetector = MotionDetector(1000, log)
        self.delta_count = 0 # difference between current frame and previous
        self.t_delta_framebuffer = np.zeros((self.height, self.width ,3), np.uint8) # framebuffer for image differencing operations

        # frame buffer housekeeping
        self.rawframe = np.zeros((self.height, self.width ,3), np.uint8) # empty img for initial diff
        (self.grabbed, self.frame) = self.stream.read() # initial frame to prime the queue
        self.frame = self.transpose(self.frame) # alignment correction

        # logging
        self.log = log # this contains reference to hud logging function in rem.py
        self.monitor_msg = '*' #  for HUD


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

            # motion detection
            self.t_delta_framebuffer = cv2.subtract(img, self.rawframe)
            self.t_delta_framebuffer = cv2.cvtColor(self.t_delta_framebuffer, cv2.COLOR_RGB2GRAY)
            _, self.t_delta_framebuffer = cv2.threshold(self.t_delta_framebuffer, 32, 255, cv2.THRESH_TOZERO)
            self.delta_count = cv2.countNonZero(self.t_delta_framebuffer)

            # dont process motion detection if paused
            if self.motiondetector.is_paused == False:
                self.motiondetector.process(self.delta_count)

            # update internal buffers w camera frame
            self.rawframe = img # unprocessed camera img
            self.frame = self.gamma_correct(self.transpose(img)) # processed camera img

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

    def __init__(self, floor, log):

        self.wasMotionDetected = False
        self.wasMotionDetected_history = False
        self.delta_count_history = 0

        self.delta_trigger = 0
        self.delta_trigger_history = 0
        self.delta_count = 0
        self.is_paused = False
        self.floor = floor
        self.update_hud_log = log
        self.history = []
        self.history_queue_length = 50
        self.forced = False
        self.monitor_msg = '*'

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

    def process(self, delta_count):
        # history
        self.delta_count = delta_count
        self.wasMotionDetected_history = self.wasMotionDetected

        # scale delta trigger so it rides peak values to prevent sensitive triggering
        self.delta_trigger = self.add_to_history(delta_count*2) + (self.floor*2)

        # logging
        # create local copies of this info
        # because it gets changed after comparison
        # and we need to see the values are were used for comparison
        _elapsed = self.elapsed
        _delta_count_history = self.delta_count_history
        _delta_trigger = self.delta_trigger

        # motion detection
        # if (self.delta_count >= self.delta_trigger and
        #     self.delta_count_history >= self.delta_trigger):

            # logging
            # reseting delta count here, for good reasons but not sure why. Possibly to force the current & previous values to be very different?
            #self.delta_count = 0

        if (self.delta_count >= self.delta_trigger and self.delta_count_history < self.delta_trigger):
            self.delta_count -= int(self.delta_count/2)
            self.update_hud_log('detect','*')
            self.wasMotionDetected = True
            self.monitor_msg += ' movement detected'
            log.critical('movement detected')



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

        self.elapsed = time.time() - self.now # elapsed time for logging function
        if self.elapsed > 5 and self.elapsed < 6:
            self.counted += 1

        # # logging
        # # preprocess self.wasMotionDetected to appear as 1/0 in datafile
        b_condition = 0
        if self.wasMotionDetected:
            b_condition = 1

        # ### logging
        # ### export data to previously defined textfile
        self.export.write('%f,%d,%d,%d,%d\n'%(
            _elapsed,
            self.delta_count,
            _delta_count_history,
            _delta_trigger,
            b_condition
            ))

        threadlog.critical('delta_trigger;{}'.format(self.delta_trigger))

        self._counter_ += 1 # used to index delta_count_history


        lastmsg = '{:0>6}'.format(self.delta_count_history)
        if self.delta_count_history > self.delta_trigger:
            # ratio = 1.0 * self.delta_count_history/self.delta_trigger
            lastmsg = '{:0>6}'.format(self.delta_count_history)

        nowmsg = '{:0>6}'.format(self.delta_count)
        if self.delta_count > self.delta_trigger:
            # ratio = 1.0 * self.delta_count/self.delta_trigger
            nowmsg = '{:0>6}'.format(self.delta_count)


        self.update_hud_log('last',lastmsg)
        self.update_hud_log('now',nowmsg)

        self.delta_count_history = self.delta_count


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


# --------
# INIT.
# --------

# setup system logging facilities
logging.config.dictConfig(LogSettings.LOGGING_CONFIG)
log = logging.getLogger('logtest-debug')
log.setLevel(logging.INFO)
threadlog = logging.getLogger('logtest-debug-thread')
threadlog.setLevel(logging.INFO)

'''
Camera Manager collects any Camera Objects
'''

# log.debug('*debug message!')
# log.info('*info message!')
# log.error('*error message')
# log.warning('warning message')
# log.critical('critical message')


