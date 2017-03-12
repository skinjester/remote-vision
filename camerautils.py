import numpy as np
import cv2

class MotionDetector(object):

    def __init__(self, delta_trigger, which_camera, log):
        self.delta_trigger = delta_trigger
        self.delta_trigger_history = delta_trigger
        self.delta_count = 0
        self.delta_count_history = 0
        self.camera = which_camera
        self.t_minus = self.camera.read()[1] 
        self.t_now = self.camera.read()[1]
        self.t_plus = self.camera.read()[1]
        self.t_delta_framebuffer = np.zeros((self.camera.get(4), self.camera.get(3) ,3), np.uint8) # empty img
        self.width = self.camera.get(3)
        self.height = self.camera.get(4)
        self.wasMotionDetected = False
        self.wasMotionDetected_history = False
        self.is_paused = False
        self.floor = 2000 # changed to lover value at night
        self.update_log = log
        self.history = []
        self.history_queue_length = 50
    
    def delta_images(self,t0, t1, t2):
        return cv2.absdiff(t2, t0)
    
   
    def process(self):
        if self.is_paused:
            self.wasMotionDetected = False
            return
        print '-'*20    
        print "[motiondetector][process]"
        # history 
        self.wasMotionDetected_history = self.wasMotionDetected  #??
        self.delta_count_history = self.delta_count   

        self.t_delta_framebuffer = self.delta_images(self.t_minus, self.t_now, self.t_plus) 
        #self.t_delta_framebuffer = cv2.flip(self.t_delta_framebuffer, 1)
        retval, self.t_delta_framebuffer = cv2.threshold(self.t_delta_framebuffer, 16, 255, 3)
        cv2.normalize(self.t_delta_framebuffer, self.t_delta_framebuffer, 0, 255, cv2.NORM_MINMAX)
        img_count_view = cv2.cvtColor(self.t_delta_framebuffer, cv2.COLOR_RGB2GRAY)
        self.delta_count = cv2.countNonZero(img_count_view)
        
        
        self.delta_trigger = self.add_to_history(self.delta_count) + self.floor
        #print 'avg:raw {}:{}'.format(self.delta_trigger, self.delta_count)

        
        
        if (self.delta_count >= self.delta_trigger and 
            self.delta_count_history >= self.delta_trigger):
            print "[motiondetector] overflow now:{} last:{}".format(self.delta_count,self.delta_count_history)
            self.delta_count = 0

        if (self.delta_count >= self.delta_trigger and self.delta_count_history < self.delta_trigger):
            self.delta_count -= int(self.delta_count/2)
            self.update_log('detect','*')
            self.wasMotionDetected = True
            print "[motiondetector] movement started"

        elif (self.delta_count < self.delta_trigger and self.delta_count_history >= self.delta_trigger):
            self.wasMotionDetected = False
            self.update_log('detect','-')
            print "[motiondetector] movement ended"
        else:
            self.update_log('detect','-')
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
        
        self.update_log('last',lastmsg)
        self.update_log('now',nowmsg)
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
        self.t_plus = self.camera.read()[1]
        self.t_plus = cv2.blur(self.t_plus,(20,20))