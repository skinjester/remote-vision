import numpy as np
import cv2

class MotionDetector(object):

    def __init__(self, delta_trigger, which_camera, log):
        self.delta_trigger = delta_trigger
        self.delta_trigger_old = delta_trigger
        self.delta_count = 0
        self.delta_count_old = 0
        self.camera = which_camera
        self.t_minus = self.camera.read()[1] 
        self.t_now = self.camera.read()[1]
        self.t_plus = self.camera.read()[1]
        self.t_delta = np.zeros((self.camera.get(4), self.camera.get(3) ,3), np.uint8) # empty img
        self.width = self.camera.get(3)
        self.height = self.camera.get(4)
        self.isMotionDetected = False
        self.isMotionDetected_old = False
        self.is_paused = False
        self.noise_level = 0
        self.update_log = log
    
    def delta_images(self,t0, t1, t2):
        return cv2.absdiff(t2, t0)
    
    def repopulate_queue(self):
        print '[motiondetector] repopulate queue'
        self.t_minus = self.camera.read()[1] 
        self.t_now = self.camera.read()[1]
        self.t_plus = self.camera.read()[1]
    
    def process(self):
        self.t_delta = self.delta_images(self.t_minus, self.t_now, self.t_plus) 
        self.t_delta = cv2.flip(self.t_delta, 1)
        retval, self.t_delta = cv2.threshold(self.t_delta, 16, 255, 3)
        cv2.normalize(self.t_delta, self.t_delta, 0, 255, cv2.NORM_MINMAX)
        img_count_view = cv2.cvtColor(self.t_delta, cv2.COLOR_RGB2GRAY)
        self.delta_count = cv2.countNonZero(img_count_view) - self.noise_level
        
        if (self.delta_count >= self.delta_trigger and self.delta_count_old >= self.delta_trigger):
            print "!!!! [motiondetector] reset now:{} last:{}".format(self.delta_count,self.delta_count_old)
            self.delta_count = 0

        self.isMotionDetected_old = self.isMotionDetected  #??
        
        if (self.delta_count >= self.delta_trigger and self.delta_count_old < self.delta_trigger):
            #self.update_log('detect','*')
            print "---- [motiondetector] movement started"
            self.isMotionDetected = True

        elif (self.delta_count < self.delta_trigger and self.delta_count_old >= self.delta_trigger):
            #update_log('detect','-')
            print "---- [motiondetector] movement ended"
            self.isMotionDetected = False
 
        else:
            #self.update_log('detect','-')
            print "---- [motiondetector] none"
            self.isMotionDetected = False

        # logging
        lastmsg = '{:0>6}'.format(self.delta_count_old)
        if self.delta_count_old > self.delta_trigger:
            ratio = 1.0 * self.delta_count_old/self.delta_trigger
            lastmsg = '{:0>6}({:02.3f})'.format(self.delta_count_old,ratio)
        
        nowmsg = '{:0>6}'.format(self.delta_count)
        if self.delta_count > self.delta_trigger:
            ratio = 1.0 * self.delta_count/self.delta_trigger
            nowmsg = '{:0>6}({:02.3f})'.format(self.delta_count,ratio)
        
        #self.update_log('last',lastmsg)
        #self.update_log('now',nowmsg)
        self.refresh_queue()

    def isResting(self):
        return self.isMotionDetected == self.isMotionDetected_old

    def refresh_queue(self):
        
        self.delta_count_old = self.delta_count   
        self.t_minus = self.t_now
        self.t_now = self.t_plus
        self.t_plus = self.camera.read()[1]
        self.t_plus = cv2.blur(self.t_plus,(16,16))