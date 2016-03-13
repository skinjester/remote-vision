#!/usr/bin/python
__author__ = 'Gary Boodhoo'

# TODO: not needing all of these imports. cleanup
import os, os.path
import argparse
import sys
import errno
import time
from random import randint
from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image 
from google.protobuf import text_format
import cv2
import data
import tweepy

os.environ['GLOG_minloglevel'] = '2'    # suppress verbose caffe logging before caffe import
import caffe

# GRB: why not just define the neural network here instead?
net = None # will become global reference to the network model once inside the loop

# HUD
# dictionary contains the key/values we'll be logging
font = cv2.FONT_HERSHEY_SIMPLEX
white = (255,255,255)

log = {}


# set global camera object to input dimensions
cap = cv2.VideoCapture(0)
cap.set(3,data.capture_size [0])
cap.set(4,data.capture_size [1])


class Amplifier(object):
    def __init__(self):
        self.iterations = None
        self.stepsize = None
        self.octaves = None
        self.octave_cutoff = None
        self.octave_scale = None
        self.iteration_mult = None
        self.step_mult = None
        self.jitter = 320
        self.package_name = None
        
    def set_package(self,key):
        self.iterations = data.settings[key]['iterations']
        self.stepsize = data.settings[key]['step_size']
        self.octaves = data.settings[key]['octaves']
        self.octave_cutoff = data.settings[key]['octave_cutoff']
        self.octave_scale = data.settings[key]['octave_scale']
        self.iteration_mult = data.settings[key]['iteration_mult']
        self.step_mult = data.settings[key]['step_mult']
        self.package_name = key


class Model(object):
    def __init__(self):
        self.guide_features = None
        self.net = None
        self.net_fn = None
        self.param_fn = None
        self.caffemodel = None
        self.end = None
        self.models = data.models
        self.guides = data.guides
        self.current_guide = 0
        self.choose_model()

    def choose_model(self, key = 'googlenet'):
        self.net_fn = '{}/{}/{}'.format(self.models['path'],self.models[key][0][0],self.models[key][0][1])
        self.param_fn = '{}/{}/{}'.format(self.models['path'],self.models[key][0][0],self.models[key][0][2])
        self.caffemodel = self.models[key][0][2]

        # Patch model to be able to compute gradients.
        model = caffe.io.caffe_pb2.NetParameter()       # load the empty protobuf model
        text_format.Merge(open(self.net_fn).read(), model)   # load the prototxt and place it in the empty model
        model.force_backward = True                     # add the force backward: true value
        open('tmp.prototxt', 'w').write(str(model))     # save it to a new file called tmp.prototxt

        # the neural network model
        self.net = caffe.Classifier('tmp.prototxt', self.param_fn,
            mean = np.float32([104.0, 116.0, 122.0]),   # ImageNet mean, training set dependent
            channel_swap = (2,1,0))  

    def guide_image(self):
        guide = np.float32(PIL.Image.open(self.guides[self.current_guide]))
        h, w = guide.shape[:2]
        src, dst = self.net.blobs['data'], self.net.blobs[self.end]
        src.reshape(1,3,h,w)
        src.data[0] = preprocess(self.net, guide)
        self.net.forward(end=self.end)
        self.guide_features = dst.data[0].copy()
        Tracker.isMotionDetected = True

    def next_guide(self):
        self.current_guide += 1
        if self.current_guide > len(self.guides)-1:
            self.current_guide = 0
        self.guide_image()

    def prev_guide(self):
        self.current_guide -= 1
        if self.current_guide < 0:
            self.current_guide = len(self.guides)-1
        self.guide_image()

    def set_endlayer(self,end):
        self.end = end
        self.guide_image()



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

        # logging
        lastmsg = '{:0>6}'.format(self.delta_count_last)
        if self.delta_count_last > self.delta_count_threshold:
            ratio = 1.0 * self.delta_count_last/self.delta_count_threshold
            lastmsg = '{:0>6}({:02.3f})'.format(self.delta_count_last,ratio)
        
        nowmsg = '{:0>6}'.format(self.delta_count)
        if self.delta_count > self.delta_count_threshold:
            ratio = 1.0 * self.delta_count/self.delta_count_threshold
            nowmsg = '{:0>6}({:02.3f})'.format(self.delta_count,ratio)
        
        update_log('last',lastmsg)
        update_log('now',nowmsg)
        self.refresh_queue()

    def isResting(self):
        return self.isMotionDetected == self.isMotionDetected_last

    def refresh_queue(self):
        self.delta_count_last = self.delta_count    
        self.t_minus = self.t_now
        self.t_now = self.t_plus
        self.t_plus = cap.read()[1]
        self.t_plus = cv2.blur(self.t_plus,(8,8))
    
class Viewport(object):

    def __init__(self, window_name='new', viewport_w=1920, viewport_h=1080, username='@skinjester'):
        self.window_name = window_name
        self.viewport_w = viewport_w
        self.viewport_h = viewport_h
        self.b_show_HUD = False
        self.motiondetect_log_enabled = False
        self.blend_ratio = 0.0
        self.save_next_frame = False
        self.username = username
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
    
    def show(self, image):
        # image is expected to be int/float array with shape (row,col,RGB)
        # convert and clip floating point matrix into RGB bounds
        image = np.uint8(np.clip(image, 0, 255))

        # GRB: check image size and skip resize if already at full size
        image = cv2.resize(image, (data.viewport_size[0], data.viewport_size[1]), interpolation = cv2.INTER_CUBIC)
        image = Frame.update(image)
        image = self.postfx(image) # HUD
        cv2.imshow(self.window_name, image)
        self.listener(image) # refresh display

    def export(self, image):
        self.save_next_frame = True
        '''
        make_sure_path_exists(self.username)
        export_path = '{}/{}.jpg'.format(self.username,time.time())
        savefile = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        PIL.Image.fromarray(np.uint8(savefile)).save(export_path)
        tweet(export_path)
        '''

    def postfx(self, image):
        if self.b_show_HUD:
            image = show_HUD(image)
        return image
    
    def monitor(self):
        if self.motiondetect_log_enabled:
            cv2.imshow('delta', Tracker.delta_view)
    
    def listener(self, image): # yeah... passing image as a convenience
        self.monitor()
        key = cv2.waitKey(1) & 0xFF
        #print '[listener] key:{}'.format(key)

        # Escape key: Exit
        if key == 27:
            self.shutdown()
            print '[listener] shutdown'

        # ENTER key: save picture
        elif key==13:
            print '[listener] save'
            self.export(image)

        # `(tilde) key: toggle HUD
        elif key == 96:
            self.b_show_HUD = not self.b_show_HUD
            print '[listener] HUD: {}'.format(Tracker.delta_count_threshold)

        # + key : increase motion threshold
        elif key == 43:
            Tracker.delta_count_threshold += 1000
            print '[listener] delta_count_threshold ++ {}'.format(Tracker.delta_count_threshold)

        # - key : decrease motion threshold    
        elif key == 45: 
            Tracker.delta_count_threshold -= 1000
            if Tracker.delta_count_threshold < 1:
                Tracker.delta_count_threshold = 1
            print '[listener] delta_count_threshold -- {}'.format(Tracker.delta_count_threshold)

        # , key : previous guide image    
        elif key == 44: 
            Dreamer.prev_guide()

        # . key : next guide image    
        elif key == 46: 
            Dreamer.next_guide()

        # 1 key : toggle motion detect window
        elif key == 49: 
            self.motiondetect_log_enabled = not self.motiondetect_log_enabled
            if self.motiondetect_log_enabled:
                cv2.namedWindow('delta',cv2.WINDOW_AUTOSIZE)
            else:
                cv2.destroyWindow('delta')   
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
        self.buffer1 = np.zeros((data.capture_size[1], data.capture_size[0] ,3), np.uint8) # uses camera capture dimensions
        self.buffer2 = np.zeros((data.viewport_size[1], data.viewport_size[0], 3), np.uint8) # uses camera capture dimensions
        self.opacity = 1.0
        self.is_compositing_enabled = False

    def update(self, image):
        s = 0.05
        if self.is_dirty: 
            print '[framebuffer] recycle'
            if self.is_new_cycle:
                # we only transform at beginning of rem cycle
                print '[framebuffer] attract fx'
                image = nd.affine_transform(image, [1-s,1,1], [data.capture_size[1]*s/2,0,0], order=1)
                self.buffer1 = image
            self.is_dirty = False
            self.is_compositing_enabled = False
        else:
            print '[framebuffer] refresh'
            if self.is_new_cycle and Tracker.isResting() == False:
                print '[framebuffer] compositing enabled'
                self.is_compositing_enabled = True
            if self.is_compositing_enabled:
                print '[framebuffer] compositing buffer1:{} buffer2:{}'.format(image.shape,self.buffer2.shape)
                image = cv2.addWeighted(self.buffer2, self.opacity, image, 1-self.opacity, 0, image)
                self.opacity -= 0.051
                if self.opacity <= 0.0:
                    self.opacity = 1.0
                    self.is_compositing_enabled = False
                    print '[framebuffer] stopped compositing'
        return image

    def write_buffer2(self,image):
        # buffer 2 is locked when compositing is enabled
        if self.is_compositing_enabled == False:
            # convert and clip floating point matrix into RGB bounds
            self.buffer2 = np.uint8(np.clip(image, 0, 255))
            ### resize buffer 2 to match viewport dimensions
            self.buffer2 = cv2.resize(self.buffer2, (data.viewport_size[0], data.viewport_size[1]), interpolation = cv2.INTER_CUBIC)
            print '[write_buffer2] copy net blob to buffer2'
        return

def make_sure_path_exists(directoryname):
    try:
        os.makedirs(directoryname)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def tweet(path_to_image):
    consumer_key='3iSUitN4D5Fi52fgmF5zMQodc'
    consumer_secret='Kj9biRwpjCBGQOmYJXd9xV4ni68IO99gZT2HfdHv86HuPhx5Mq'
    access_key='15870561-2SH025poSRlXyzAGc1YyrL8EDgD5O24docOjlyW5O'
    access_secret='qwuc8aa6cpRRKXxMObpaNhtpXAiDm6g2LFfzWhSjv6r8H'
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    fn = os.path.abspath('../eagle.jpg')
    #myStatusText = '@username #deepdreamvisionquest #GDC2016'
    myStatusText = '{} #deepdreamvisionquest #test bit.ly/1Rfj3gN'.format(Viewer.username)
    api.update_with_media(path_to_image, status=myStatusText )

def update_log(key,new_value):
    log[key] = '{}'.format(new_value)

def show_HUD(image):
    # rectangle
    overlay = image.copy()
    opacity = 0.5
    cv2.rectangle(overlay,(0,0),(840,data.viewport_size[1]),(0,0,0),-1)

    # list setup
    x,xoff = 40,240
    y,yoff = 150,35

    data.counter = 0
    def write_Text(key):
    	row = y + yoff * data.counter
        cv2.putText(overlay, key, (x, row), font, 1.0, white)
        cv2.putText(overlay, log[key], (xoff, row), font, 1.0, white)
        data.counter += 1

    # write text to overlay
    # col1
    cv2.putText(overlay, log['detect'], (5, 35), font, 2.0, (0,255,0))
    cv2.putText(overlay, 'DEEPDREAMVISIONQUEST', (x, 100), font, 2.0, white)
    write_Text('username')
    write_Text('settings')
    write_Text('threshold')
    write_Text('last')
    write_Text('now')
    write_Text('model')
    write_Text('layer')
    write_Text('guide')
    write_Text('width')
    write_Text('height')
    write_Text('octave')
    write_Text('iteration')
    write_Text('step_size')


    #col2

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
    print '[objective_guide] update dream features'
    x = dst.data[0].copy()
    y = Dreamer.guide_features
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
    src = Dreamer.net.blobs['data']
    octaves = [preprocess(Dreamer.net, base_img)]
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
            make_step(Dreamer.net, end=end, clip=clip, **step_params)
            print '{:02d}:{:03d}:{:03d}'.format(octave,i,iter_n)

            # output - deprocess net blob and write to frame buffer
            Frame.buffer1 = deprocess(Dreamer.net, src.data[0])
            Frame.buffer1 = Frame.buffer1 * (255.0 / np.percentile(Frame.buffer1, 99.98)) # normalize contrast
            Tracker.process()
            Viewer.show(Frame.buffer1)

            # attenuate step size over rem cycle
            x = step_params['step_size']
            step_params['step_size'] += x * Amplify.step_mult

            i += 1

            # logging
            octavemsg = '{}/{}({})'.format(len(octaves) - octave - 1,octave_n,Amplify.octave_cutoff)
            guidemsg = '({}/{}) {}'.format(Dreamer.current_guide,len(Dreamer.guides),Dreamer.guides[Dreamer.current_guide])
            iterationmsg = '{:0>3}/{:0>3}({})'.format(i,iter_n,Amplify.iteration_mult)
            stepsizemsg = '{:02.3f}({:02.3f})'.format(step_params['step_size'],Amplify.step_mult)
            thresholdmsg = '{:0>6}'.format(Tracker.delta_count_threshold)
            update_log('octave',octavemsg)
            update_log('width',w)
            update_log('height',h)
            update_log('guide',guidemsg)
            update_log('layer',end)
            update_log('iteration',iterationmsg)
            update_log('step_size',stepsizemsg)
            update_log('settings',Amplify.package_name)
            update_log('threshold',thresholdmsg)


        # early return this will be the last octave calculated in the series
        if octave == Amplify.octave_cutoff:
            Frame.is_dirty = True
            early_exit = deprocess(Dreamer.net, src.data[0])
            early_exit = cv2.resize(early_exit, (data.capture_size[0], data.capture_size[1]), interpolation = cv2.INTER_CUBIC)
            print '[deepdream] {:02d}:{:03d}:{:03d} early return net blob'.format(octave,i,iter_n)
            return early_exit

        # motion detected so we're ending this REM cycle
        if Tracker.isMotionDetected:
            early_exit = deprocess(Dreamer.net, src.data[0])  # pass deprocessed net blob to buffer2 for fx
            early_exit = cv2.resize(early_exit, (Viewer.viewport_w, Viewer.viewport_h), interpolation = cv2.INTER_CUBIC) # normalize size to match camera input
            Frame.write_buffer2(early_exit)
            Frame.is_dirty = False # no, we'll be refreshing the frane buffer
            print '[deepdream] return camera'
            return cap.read()[1] 

        # reduce iteration count for the next octave
        detail = src.data[0] - octave_base # extract details produced on the current octave
        iter_n = iter_n - int(iter_n * Amplify.iteration_mult)
        #iter_n = Amplify.next_iteration(iter_n)

    # return the resulting image (converted back to x,y,RGB structured matrix)
    print '[deepdream] {:02d}:{:03d}:{:03d} return net blob'.format(octave,i,iter_n)
    Frame.is_dirty = True # yes, we'll be recycling the framebuffer
    return deprocess(Dreamer.net, src.data[0])


# -------
# MAIN
# ------- 
def main():

    # start timer
    data.now = time.time()

    # set GPU mode
    caffe.set_device(0)
    caffe.set_mode_gpu()

    Dreamer.choose_model('googlenet')
    Dreamer.set_endlayer(data.layers[0])

    # parameters
    Amplify.set_package('hifi')
    iterations = Amplify.iterations
    stepsize = Amplify.stepsize
    octaves = Amplify.octaves
    octave_scale = Amplify.octave_scale
    jitter = 300
    update_log('model',Dreamer.caffemodel)
    update_log('username',Viewer.username)

    # the madness begins 
    Frame.buffer1 = cap.read()[1] # initial camera image for init
    while True:
        Frame.is_new_cycle = True
        Viewer.show(Frame.buffer1)
        Tracker.process()

        # kicks off rem sleep - will begin continual iteration of the image through the model
        Frame.buffer1 = deepdream(net, Frame.buffer1, iter_n = iterations, octave_n = octaves, octave_scale = octave_scale, step_size = stepsize, end = Dreamer.end )

        if Viewer.save_next_frame:
            print '[main] save rendered frame'
            Viewer.save_next_frame = False
            make_sure_path_exists(Viewer.username)
            export_path = '{}/{}.jpg'.format(Viewer.username,time.time())
            savefile = cv2.cvtColor(Frame.buffer1, cv2.COLOR_BGR2RGB)
            PIL.Image.fromarray(np.uint8(savefile)).save(export_path)
            tweet(export_path)

        # a bit later
        later = time.time()
        difference = int(later - data.now)
        print '[main] finish REM cycle:{}s'.format(difference)
        print '-'*20

        data.now = time.time()


# -------- 
# INIT
# --------
Tracker = MotionDetector(10000)
Viewer = Viewport('deepdreamvisionquest',1920,1080,'@skinjester')
Frame = Framebuffer()
Dreamer = Model()
Amplify = Amplifier()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--username',help='twitter userid for sharing')
    args = parser.parse_args()
    if args.username:
        Viewer.username = '@{}'.format(args.username)
    main()
