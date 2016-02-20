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

# viewport
viewport_w, viewport_h = 1280,720 # display resolution
b_debug = False
font = cv2.FONT_HERSHEY_PLAIN
white = (255,255,255)

# camera object
cap = cv2.VideoCapture(0)
cap_w, cap_h = 1280,720 # capture resolution
cap.set(3,cap_w)
cap.set(4,cap_h)

# motion detection - prepopulate queue before we enter the loop
t_minus = cap.read()[1]
t_now = cap.read()[1]
t_plus = cap.read()[1]
delta_count_last = 1
record_video_state = False
DELTA_COUNT_THRESHOLD = 80000

# dictionary for the values we'll be logging
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
    'detect':'*'
}

# -------
# utility
# ------- 

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
    cv2.rectangle(overlay,(0,0),(viewport_w,240),(0,0,0),-1)

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
    cv2.putText(overlay, log['detect'], (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0))

    # add overlay back to source
    cv2.addWeighted(overlay, opacity, image, 1-opacity, 0, image)


# this creates a RGB image from our image matrix
# GRB: what is the expected input here??
# GRB: why do I have 3 functions (below) to write images to the display?
def showarray(window_name, a):
    global b_debug, DELTA_COUNT_THRESHOLD

    # convert and clip our floating point matrix into 0-255 values for RGB image
    a = np.uint8(np.clip(a, 0, 255))

    # resize takes its arguments as w,h in that order
    dim = (viewport_w, viewport_h)
    a = cv2.resize(a, dim, interpolation = cv2.INTER_LINEAR)

    if b_debug:
        show_HUD(a)

    # write to window
    cv2.imshow(window_name, a)

    # refresh the display 
    key = cv2.waitKey(1) & 0xFF
    if key == 27: # Escape key: Exit
        sys.exit()
    elif key == 96: # `(tilde) key: toggle HUD
        b_debug = not b_debug
    elif key == 43: # + key : increase motion threshold
        DELTA_COUNT_THRESHOLD += 100
    elif key == 45: # - key : decrease motion threshold
        DELTA_COUNT_THRESHOLD -= 100
        if DELTA_COUNT_THRESHOLD < 1:
            DELTA_COUNT_THRESHOLD = 1

# writes opencv img to window and updates display
def showimg(window_name, a):
    cv2.imshow(window_name, a)
    key = cv2.waitKey(1)

# GRB: don't the preprocess/deprocess functions already do this?
def showcaffe(signal_name, caffe_array):
    # convert caffe format to Row,Col,RGB array for visualizing
    vis = deprocess(net, caffe_array)
    vis = vis * (255.0 / np.percentile(vis, 100.0))
    showarray(signal_name,vis)
    return vis

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    #print np.float32(img).shape
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def objective_L2(dst):
    dst.diff[:] = dst.data



# -------
# motion detector utility functions
# ------- 
def delta_images(t0, t1, t2):
    d1 = cv2.absdiff(t2, t0)
    return d1
    

# -------
# implements forward and backward passes thru the network
# apply normalized ascent step upon the image in the networks data blob
# ------- 
def make_step(net, step_size=1.5, end='inception_4c/output',jitter=32, clip=True):
    src = net.blobs['data']     # input image is stored in Net's 'data' blob
    dst = net.blobs[end]        # destination is the end layer specified by argument

    ox, oy = np.random.randint(-jitter, jitter + 1, 2)          # calculate jitter
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift

    # this bit is where the neural net runs the computation

    net.forward(end=end)    # make sure we stop on the chosen neural layer
    dst.diff[:] = dst.data  # specify the optimization objective
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
# iterates ththru the neural network 
# REM sleep, in other words
# ------- 
def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, end='inception_4c/output', clip=True, **step_params):

    global t_now,t_minus,t_plus,delta_count_last,record_video_state,cap

    
    src = net.blobs['data']

    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0 / octave_scale, 1.0 / octave_scale), order=1))
    detail = np.zeros_like(octaves[-1])

    # REM cycle on octaves
    for octave, octave_base in enumerate(octaves[::-1]):
        
        h, w = octave_base.shape[-2:]
        if octave > 0:
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0 * h / h1, 1.0 * w / w1), order=0)

        src.reshape(1,3,h,w)
        src.data[0] = octave_base + detail

        # logging

        # iterate on current octave
        for i in xrange(iter_n):

            # motion detect
            delta_view = delta_images(t_minus, t_now, t_plus)
            retval, delta_view = cv2.threshold(delta_view, 16, 255, 3)
            cv2.normalize(delta_view, delta_view, 0, 255, cv2.NORM_MINMAX)
            img_count_view = cv2.cvtColor(delta_view, cv2.COLOR_RGB2GRAY)
            delta_count = cv2.countNonZero(img_count_view)
            delta_view = cv2.flip(delta_view, 1)

            # logging
            update_log('octave',len(octaves) - octave - 1)
            update_log('width',w)
            update_log('height',h)
            update_log('pixels',w*h)
            update_log('layer',end)
            update_log('iteration',i)
            update_log('threshold',DELTA_COUNT_THRESHOLD)
            update_log('last',delta_count_last)
            update_log('now',delta_count)
            update_log('ratio',1.0 * DELTA_COUNT_THRESHOLD/(w * h)*100)

            if (delta_count_last < DELTA_COUNT_THRESHOLD and delta_count >= DELTA_COUNT_THRESHOLD):
                record_video_state = True
                update_log('detect','*')
                print "+ MOVEMENT DETECTED"
            elif delta_count_last >= DELTA_COUNT_THRESHOLD and delta_count < DELTA_COUNT_THRESHOLD:
                record_video_state = False
                update_log('detect','')
                print "+ MOVEMENT CLEARED"

            # move images through the motion detect queue.
            delta_count_last = delta_count
            t_minus = t_now
            t_now = t_plus
            t_plus = cap.read()[1]
            t_plus = cv2.blur(t_plus,(8,8))
            t_plus = cv2.resize(t_plus, (cap_w, cap_h)) # necessary to resize?

            if record_video_state == True:
                print '+ TERMINATE ITERATION'
                break

            # calls the neural net step function
            make_step(net, end=end, clip=clip, **step_params)

            # output
            showcaffe('new',src.data[0])
            #print '[deepdream] octave:{:3} iteration:{:3} end:{:} shape:{:<10}'.format(octave, i, end, src.data[0].shape)


        # extract details produced on the current octave
        detail = src.data[0] - octave_base

    # return the resulting image (converted back to x,y,RGB structured matrix)
    if record_video_state:
        print '[deepdream] return new camera img from loopreset'
        return cap.read()[1]
    else:
        print '[deepdream] return RGB from net blob'
        return deprocess(net, src.data[0])


# -------
# MAIN
# ------- 
def main(iterations, stepsize, octaves, octave_scale, end):
    global net

    # start timer
    print '+ TIMER START :REM.main'
    now = time.time()

    # set GPU mode
    caffe.set_device(0)
    caffe.set_mode_gpu()

    cv2.startWindowThread()
    cv2.namedWindow('new',cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow('input',cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow('deltaview',cv2.WINDOW_AUTOSIZE)

    # parameters
    model_path = 'E:/Users/Gary/Documents/code/models/cars/'
    net_fn = model_path + 'deploy.prototxt'
    param_fn = model_path + 'googlenet_finetune_web_car_iter_10000.caffemodel'

    nrframes = 1
    jitter = int(cap_w/2)
    zoom = 1

    if iterations is None: iterations = 10
    if stepsize is None: stepsize = 2
    if octaves is None: octaves = 4
    if octave_scale is None: octave_scale = 1.5
    if end is None: end = 'inception_5a_pool'

    print '[main] iterations:{arg1} step size:{arg2} octaves:{arg3} octave_scale:{arg4} end:{arg5}'.format(arg1=iterations,arg2=stepsize,arg3=octaves,arg4=octave_scale,arg5=end)

    # Patching model to be able to compute gradients.
    model = caffe.io.caffe_pb2.NetParameter()       # load the empty protobuf model
    text_format.Merge(open(net_fn).read(), model)   # load the prototxt and place it in the empty model
    model.force_backward = True                     # add the force backward: true value
    open('tmp.prototxt', 'w').write(str(model))     # save it to a new file called tmp.prototxt

    # the neural network model
    net = caffe.Classifier('tmp.prototxt', param_fn,
        mean = np.float32([104.0, 116.0, 122.0]),   # ImageNet mean, training set dependent
        channel_swap = (2,1,0))                     # the caffe reference model has chanels in BGR instead of RGB


    frame = cap.read()[1] # initial camera image for init
    s = 0.05 # scale coefficient for uninterrupted dreaming
    while True:
        # zoom in a bit on the frame
        frame = frame*(255.0/np.percentile(cap.read()[1], 98))
        frame = nd.affine_transform(frame, [1-s,1-s,1], [cap_h*s/2,cap_w*s/2,0], order=1)

        showarray('new',frame)

        # a bit later
        later = time.time()
        difference = int(later - now)

        # a bit later
        later = time.time()
        difference = int(later - now)
        print '+ ELAPSED: {}s :{}'.format(difference,'start REM cycle')

        # kicks off rem sleep - will begin continual iteration of the image through the model
        frame = deepdream(net, frame, iter_n = iterations, octave_n = octaves, octave_scale = octave_scale, step_size = stepsize, end = end)

        # a bit later
        later = time.time()
        difference = int(later - now)
        print '+ ELAPSED: {}s :{}'.format(difference,'finish REM cycle')



# ------- 
# INIT
# ------- 
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
    parser.add_argument(
        '-c','--clip',
        type=float,
        required=False,
        help='Step Size. Default: 1.5')
    args = parser.parse_args()
    main(args.iterations, args.stepsize, args.octaves,  args.octavescale, args.end)
