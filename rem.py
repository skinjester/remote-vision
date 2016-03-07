#!/usr/bin/python
__author__ = 'Gary Boodhoo'

# TODO: not needing all of these imports. cleanup
import argparse
import os, os.path
import re
import errno
import sys
import time
import subprocess
from random import randint
from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image 
from google.protobuf import text_format
import cv2

os.environ['GLOG_minloglevel'] = '2'    # suppress verbose caffe logging before caffe import
import caffe

# GRB: why not just define the neural network here instead?
net = None # will become global reference to the network model once inside the loop

# HUD
# dictionary is for the values we'll be logging
font = cv2.FONT_HERSHEY_PLAIN
white = (255,255,255)
log = {
    'threshold':'{:0>6}'.format(10000),
    'last':'{:0>6}'.format(43),
    'now':'{:0>6}'.format(9540),
    'ratio':'{:0>1.4f}'.format(0.5646584),
    'octave':'{}'.format(0.564),
    'pixels':'{:0>6}'.format(960*540),
    'width':'{}'.format(960),
    'height':'{}'.format(540),
    'model':'googlenet_finetune_web_car_iter_10000.caffemodel',
    'layer':'inception_4c_pool',
    'iteration':'{}'.format(10),
    'detect':'*',
    'cyclelength':'0 sec',
    'step_size':'{}'.format(0.0)
}

# global camera object
cap = cv2.VideoCapture(0)
cap_w, cap_h = 1280,720 # capture resolution
cap.set(3,cap_w)
cap.set(4,cap_h)





class MotionDetector(object):
    # cap: global capture object
    def __init__(self, delta_count_threshold=5000):
        self.delta_count_threshold = delta_count_threshold
        self.delta_count = 0
        self.delta_count_last = 0
        self.t_minus = cap.read()[1] 
        self.t_now = cap.read()[1]
        self.t_plus = cap.read()[1]
        self.width = cap.get(3)
        self.height = cap.get(4)
        self.delta_view = np.zeros((cap.get(4), cap.get(3) ,3), np.uint8) # empty img
        self.isMotionDetected = False
        self.isMotionDetected_last = False
        self.timer_enabled = False
    
    def delta_images(self,t0, t1, t2):
        d1 = cv2.absdiff(t2, t0)
        return d1
    
    def repopulate_queue(self):
        print '[motiondetector] repopulate queue'
        self.t_minus = cap.read()[1] 
        self.t_now = cap.read()[1]
        self.t_plus = cap.read()[1]
    
    def process(self):
        self.delta_view = self.delta_images(self.t_minus, self.t_now, self.t_plus) 
        retval, self.delta_view = cv2.threshold(self.delta_view, 16, 255, 3)
        cv2.normalize(self.delta_view, self.delta_view, 0, 255, cv2.NORM_MINMAX)
        img_count_view = cv2.cvtColor(self.delta_view, cv2.COLOR_RGB2GRAY)
        self.delta_count = cv2.countNonZero(img_count_view)
        self.delta_view = cv2.flip(self.delta_view, 1)
        self.isMotionDetected_last = self.isMotionDetected  
        if (self.delta_count_last < self.delta_count_threshold and self.delta_count >= self.delta_count_threshold):
            update_log('detect','*')
            print "---- [motiondetector] movement started"
            self.isMotionDetected = True
            self.timer_start = time.time()
            self.timer_enabled = True
            # start timer/start counting    
        elif (self.delta_count_last >= self.delta_count_threshold and self.delta_count < self.delta_count_threshold):
            update_log('detect',' ')
            print "---- [motiondetector] movement ended"
            self.isMotionDetected = False
            self.timer_enabled = False
            # stop timer/stop counting    
        elif self.timer_enabled:
            now = time.time()
            # sooo... we shouldn't end up here at all if the timer was properly reset
            # if timer value > n fire stop timer event
            if int(now - self.timer_start) > 8:
                update_log('detect',' ')
                print "---- [motiondetector] force movement end"
                self.isMotionDetected = False
                self.timer_enabled = False
        else:
            self.isMotionDetected = False
    
        update_log('threshold',Tracker.delta_count_threshold)
        update_log('last',Tracker.delta_count_last)
        update_log('now',Tracker.delta_count)
        self.refresh_queue()
    
    def isResting(self):
        return self.isMotionDetected == self.isMotionDetected_last
    
    def refresh_queue(self):
        self.delta_count_last = self.delta_count    
        self.t_minus = self.t_now
        self.t_now = self.t_plus
        self.t_plus = cap.read()[1]
        self.t_plus = cv2.blur(self.t_plus,(8,8))
    
    def monitor(self,isEnabled):
        # I want to do two things:
        # - if monitoring is enabled then initially create a window for the display
        # - write self.delta_view to this window
        # this method will be called from within the showarray() function
        pass

class Viewport(object):

    def __init__(self, window_name='new', viewport_w=1280, viewport_h=720):
        self.window_name = window_name
        self.viewport_w = viewport_w
        self.viewport_h = viewport_h
        self.b_show_HUD = False
        self.motiondetect_log_enabled = False
        self.image_buffer = np.zeros((cap.get(4), cap.get(3) ,3), np.uint8) # uses camera capture size
        self.blend_ratio = 0.0
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
    
    def show(self, image):
        # image is expected to be int/float array with shape (row,col,RGB)
        # convert and clip floating point matrix into RGB bounds
        image = np.uint8(np.clip(image, 0, 255))

        # GRB: check image size and skip resize if already at full size
        image = cv2.resize(image, (self.viewport_w, self.viewport_h), interpolation = cv2.INTER_CUBIC)
        image = Frame.update(image)
        image = self.postfx(image) # HUD
        cv2.imshow(self.window_name, image)
        self.listener() # refresh display

    def postfx(self, image):
        if self.b_show_HUD:
            image = show_HUD(image)
        return image
    
    def monitor(self):
        if self.motiondetect_log_enabled:
            cv2.imshow('deltaview', Tracker.delta_view)
    
    def listener(self):
        self.monitor()
        key = cv2.waitKey(1) & 0xFF

         # Escape key: Exit
        if key == 27:
            self.shutdown()
            print '[keylistener] shutdown'
        # `(tilde) key: toggle HUD
        elif key == 96:
            self.b_show_HUD = not self.b_show_HUD
            print '[keylistener] HUD: {}'.format(Tracker.delta_count_threshold)

        # + key : increase motion threshold
        elif key == 43:
            Tracker.delta_count_threshold += 1000
            print '[keylistener] delta_count_threshold ++ {}'.format(Tracker.delta_count_threshold)

        # - key : decrease motion threshold    
        elif key == 45: 
            Tracker.delta_count_threshold -= 1000
            if Tracker.delta_count_threshold < 1:
                Tracker.delta_count_threshold = 0
            print '[keylistener] delta_count_threshold -- {}'.format(Tracker.delta_count_threshold)

        # 1 key : toggle motion detect window
        elif key == 49: 
            self.motiondetect_log_enabled = not self.motiondetect_log_enabled
            if self.motiondetect_log_enabled:
                cv2.namedWindow('deltaview',cv2.WINDOW_AUTOSIZE)
            else:
                cv2.destroyWindow('delta_view')   
            print '[keylistener] motion detect monitor: {}'.format(self.motiondetect_log_enabled)

    #self.monitor() # update the monitor windows
    def show_blob(self, net, caffe_array):
        image = deprocess(net, caffe_array)
        image = image * (255.0 / np.percentile(image, 100.0))
        self.show(image)

    def shutdown(self):
        sys.exit()

class Framebuffer(object):

    def __init__(self):
        self.is_dirty = False # the type of frame in buffer1. dirty when recycling clean when refreshing
        self.is_new_cycle = True
        self.buffer1 = np.zeros((cap.get(4), cap.get(3) ,3), np.uint8) # uses camera capture dimensions
        self.buffer2 = np.zeros((cap.get(4), cap.get(3) ,3), np.uint8) # uses camera capture dimensions
        self.opacity = 1.0
        self.is_compositing_enabled = False
        self.guide_features = None

    def update(self, image):
        s = 0.05
        if self.is_dirty: 
            print '[framebuffer] recycle'
            if self.is_new_cycle:
                # we only transform at beginning of rem cycle
                print '[framebuffer] attract fx'
                image = nd.affine_transform(image, [1-s,1,1], [cap_h*s/2,0,0], order=1)
                self.buffer1 = image
            self.is_dirty = False
            self.is_compositing_enabled = False
        else:
            print '[framebuffer] refresh'
            if self.is_new_cycle and Tracker.isResting() == False:
                print '[framebuffer] compositing enabled'
                self.is_compositing_enabled = True
            if self.is_compositing_enabled:
                print '[framebuffer] compositing'
                image = cv2.addWeighted(self.buffer2, self.opacity, image, 1-self.opacity, 0, image)
                self.opacity -= 0.05
                if self.opacity <= 0.0:
                    self.opacity = 1.0
                    self.is_compositing_enabled = False
                    print '[framebuffer] stopped compositing'
        return image

    def write_buffer2(self,image):
        # buffer 2 is locked when compositing is enabled
        if self.is_compositing_enabled == False:
            # convert and clip floating point matrix into RGB bounds
            print '[write_buffer2] copy net blob to buffer2'
            self.buffer2 = np.uint8(np.clip(image, 0, 255))
        return


# GRB: if these values were all global there'd be no need to pass them explicitly to this function
def update_log(key,new_value):
    if key=='threshold':
        log[key] = '{:0>6}'.format(new_value)
    elif key=='last':
        log[key] = '{:0>6}'.format(new_value)
    elif key=='now':
        log[key] = '{:0>6}'.format(new_value)
    elif key=='ratio':
        log[key] = '{:0>1.4f}'.format(new_value)
    elif key=='pixels':
        log[key] = '{:0>6}'.format(new_value)
    else:
        log[key] = '{}'.format(new_value)

def show_HUD(image):
    # rectangle
    overlay = image.copy()
    opacity = 0.5
    cv2.rectangle(overlay,(0,0),(Viewer.viewport_w,240),(0,0,0),-1)

    # list setup
    col1,y,col2,y1 = 5,50,100,15

    def write_Text(row,subject):
        cv2.putText(overlay, subject, (col1,row), font, 1.0, white)
        cv2.putText(overlay, log[subject], (col2, row), font, 1.0, white)

    # write text to overlay
    write_Text(y, 'pixels')
    write_Text(y + y1, 'threshold')
    write_Text(y + y1 * 2, 'ratio')
    write_Text(y + y1 * 3, 'last')
    write_Text(y + y1 * 4, 'now')
    write_Text(y + y1 * 5, 'model')
    write_Text(y + y1 * 6, 'layer')
    write_Text(y + y1 * 7, 'width')
    write_Text(y + y1 * 8, 'height')
    write_Text(y + y1 * 9, 'octave')
    write_Text(y + y1 * 10, 'iteration')
    write_Text(y + y1 * 11, 'step_size')
    cv2.putText(overlay, log['detect'], (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0))

    # add overlay back to source
    return cv2.addWeighted(overlay, opacity, image, 1-opacity, 0, image)

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

def deprocess(net, img):
    a = np.dstack((img + net.transformer.mean['data'])[::-1])
    return a

def objective_L2(dst):
    dst.diff[:] = dst.data

def objective_guide(dst):
    x = dst.data[0].copy()
    y = Frame.guide_features
    ch = x.shape[0]
    x = x.reshape(ch,-1)
    y = y.reshape(ch,-1)
    A = x.T.dot(y) # compute the matrix of dot produts with guide features
    dst.diff[0].reshape(ch,-1)[:] = y[:,A.argmax(1)] # select one sthta match best




# -------
# implements forward and backward passes thru the network
# apply normalized ascent step upon the image in the networks data blob
# ------- 
def make_step(net, step_size=1.5, end='inception_4c/output',jitter=32, clip=True, objective=objective_L2):
    src = net.blobs['data']     # input image is stored in Net's 'data' blob
    dst = net.blobs[end]        # destination is the end layer specified by argument

    ox, oy = np.random.randint(-jitter, jitter + 1, 2)          # calculate jitter
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift

    # this bit is where the neural net runs the computation
    net.forward(end=end)    # make sure we stop on the chosen neural layer
    objective(dst)          # specify the optimization objective
    net.backward(start=end) # backwards propagation
    g = src.diff[0]         # store the error 

    # apply normalized ascent step to the input image and get closer to our target state
    src.data[:] += step_size / np.abs(g).mean() * g

    # unshift image jitter              
    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2)   

    # subtract image mean and clip our matrix to the values
    bias = net.transformer.mean['data']
    src.data[:] = np.clip(src.data, -bias, 255-bias)

# -------
# sets up image buffers and octave structure for iterating thru and amplifying neural output
# iterates thru the neural network 
# REM sleep, in other words
# ------- 
def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, end='inception_4c/output', clip=True, **step_params):

    # before doing anything check the current value of Tracker.isResting()
    if Tracker.isResting() == False and Tracker.isMotionDetected:
        print '[deepdream] cooldown'
        return cap.read()[1]
    print '[deepdream] new cycle'
    Frame.is_new_cycle = False

    # setup octaves
    
    src = net.blobs['data']
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0 / octave_scale, 1.0 / octave_scale), order=1))
    detail = np.zeros_like(octaves[-1])

    # REM cycle, last octave first
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        h1, w1 = detail.shape[-2:]
        detail = nd.zoom(detail, (1, 1.0 * h / h1, 1.0 * w / w1), order=0)
        src.reshape(1,3,h,w)
        src.data[0] = octave_base + detail

        i=0 # iterate on current octave
        while i < iter_n and Tracker.isMotionDetected == False:
            # delegate gradient ascent to step function
            make_step(net, end=end, clip=clip, **step_params)
            print '{:02d}:{:03d}:{:03d}'.format(octave,i,iter_n)

            # output - deprocess net blob and write to frame buffer
            Frame.buffer1 = deprocess(net, src.data[0])
            Frame.buffer1 = Frame.buffer1 * (255.0 / np.percentile(Frame.buffer1, 99.98)) # normalize contrast
            Tracker.process()
            Viewer.show(Frame.buffer1)

            # attenuate step size over rem cycle
            x = step_params['step_size']
            #step_params['step_size'] -= step_params['step_size'] * 0.1
            #step_params['step_size'] += 0.1
            step_params['step_size'] += x * 0.05

            i += 1

            # logging
            update_log('octave',len(octaves) - octave - 1)
            update_log('width',w)
            update_log('height',h)
            update_log('pixels',w*h)
            update_log('layer',end)
            update_log('iteration',i)
            update_log('step_size',step_params['step_size'])

        detail = src.data[0] - octave_base # extract details produced on the current octave
        
        # early return this will be the last octave calculated in the series
        if octave == 10:
            Frame.is_dirty = True
            early_exit = deprocess(net, src.data[0])
            early_exit = cv2.resize(early_exit, (cap_w, cap_h), interpolation = cv2.INTER_CUBIC)
            print '[deepdream] {:02d}:{:03d}:{:03d} early return net blob'.format(octave,i,iter_n)
            return early_exit

        # motion detected so we're ending this REM cycle
        if Tracker.isMotionDetected:
            early_exit = deprocess(net, src.data[0])  # pass deprocessed network blob to buffer2 for fx
            early_exit = cv2.resize(early_exit, (cap_w, cap_h), interpolation = cv2.INTER_CUBIC) # normalize size to match camera input
            Frame.write_buffer2(early_exit)
            Frame.is_dirty = False # no, we'll be refreshing the frane buffer
            print '[deepdream] return camera'
            return cap.read()[1] 

        # reduce iteration count for the next octave
        iter_n = iter_n - int(iter_n*0.5)

    # return the resulting image (converted back to x,y,RGB structured matrix)
    print '[deepdream] {:02d}:{:03d}:{:03d} return net blob'.format(octave,i,iter_n)
    Frame.is_dirty = True # yes, we'll be recycling the framebuffer
    return deprocess(net, src.data[0])


# -------
# MAIN
# ------- 
def main(iterations, stepsize, octaves, octave_scale, end):
    global net

    # start timer
    now = time.time()

    # set GPU mode
    caffe.set_device(0)
    caffe.set_mode_gpu()

    # path to model
    model_path = 'E:/Users/Gary/Documents/code/models/bvlc_googlenet/'
    net_fn = model_path + 'deploy.prototxt'
    caffemodel = 'bvlc_googlenet.caffemodel'
    param_fn = model_path + caffemodel

    # Patching model to be able to compute gradients.
    model = caffe.io.caffe_pb2.NetParameter()       # load the empty protobuf model
    text_format.Merge(open(net_fn).read(), model)   # load the prototxt and place it in the empty model
    model.force_backward = True                     # add the force backward: true value
    open('tmp.prototxt', 'w').write(str(model))     # save it to a new file called tmp.prototxt

    # the neural network model
    net = caffe.Classifier('tmp.prototxt', param_fn,
        mean = np.float32([104.0, 116.0, 122.0]),   # ImageNet mean, training set dependent
        channel_swap = (2,1,0))                     # the caffe reference model has chanels in BGR instead of RGB
    
    # parameters
    jitter = int(cap_w/2)
    if iterations is None: iterations = 10
    if stepsize is None: stepsize = 1.0
    if octaves is None: octaves = 5
    if octave_scale is None: octave_scale = 1.4
    if end is None: end = 'inception_4c/5x5'
    update_log('model',caffemodel)

    # guide image
    guide = np.float32(PIL.Image.open('tiger.jpg'))
    h, w = guide.shape[:2]
    src, dst = net.blobs['data'], net.blobs[end]
    src.reshape(1,3,h,w)
    src.data[0] = preprocess(net, guide)
    net.forward(end=end)
    Frame.guide_features = dst.data[0].copy()



    # the madness begins 
    Frame.buffer1 = cap.read()[1] # initial camera image for init
    while True:
        Frame.is_new_cycle = True
        Viewer.show(Frame.buffer1)
        Tracker.process()

        # kicks off rem sleep - will begin continual iteration of the image through the model
        Frame.buffer1 = deepdream(net, Frame.buffer1, iter_n = iterations, octave_n = octaves, octave_scale = octave_scale, step_size = stepsize, end = end, objective = objective_guide )

        # a bit later
        later = time.time()
        difference = int(later - now)
        print '[main] finish REM cycle:{}s'.format(difference)
        print '-'*20

        now = time.time()



# -------- 
# INIT
# --------
Tracker = MotionDetector(60000) # motion detector object
Viewer = Viewport() # viewport object
Frame = Framebuffer()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='REM')
    parser.add_argument(
        '-e','--end',
        required=False,
        help='End layer. Default: inception_4c/output')
    parser.add_argument(
        '-oct','--octaves',
        type=int,
        required=False,
        help='Octaves. Default: 4')
    parser.add_argument(
        '-octs','--octavescale',
        type=float,
        required=False,
        help='Octave Scale. Default: 1.4',)
    parser.add_argument(
        '-i','--iterations',
        type=int,
        required=False,
        help='Iterations. Default: 10')
    parser.add_argument(
        '-s','--stepsize',
        type=float,
        required=False,
        help='Step Size. Default: 1.5')
    args = parser.parse_args()
    main(args.iterations, args.stepsize, args.octaves,  args.octavescale, args.end)
