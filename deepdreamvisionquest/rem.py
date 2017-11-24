# rem.py

import os
import os.path
import argparse
import sys
import errno
import time
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from google.protobuf import text_format
import cv2
import data
import tweepy
os.environ['GLOG_minloglevel'] = '2' # suppress verbose caffe logging before caffe import
import caffe
from camerautils import MotionDetector
from camerautils import WebcamVideoStream
from camerautils import Cameras
from random import randint
import inspect
import logging
import logging.config
sys.path.append('../bin') #  point to directory containing LogSettings
import LogSettings # global log settings templ

import math
from collections import deque



# using this to index some values with dot notation in their own namespace
# perhaps misguided, but needing expediency at the moment
class Display(object):
    def __init__(self, width, height, camera):
        self.width = width
        self.height = height

        # swap width, height when in portrait alignment
        if camera.portrait_alignment:
            self.width = height
            self.height = width


class Model(object):
    def __init__(self, current_layer=0, program_duration=10):
        self.guide_features = None
        self.net = None
        self.net_fn = None
        self.param_fn = None
        self.caffemodel = None
        self.end = None
        self.models = data.models
        self.guides = data.guides

        self.features = None
        self.current_feature = 0

        self.current_guide = 0
        self.current_layer = current_layer
        self.layers = data.layers
        self.first_time_through = True

        self.program = data.program
        self.current_program = 0
        self.program_duration = program_duration
        self.program_start_time = time.time()

        # amplification
        self.iterations = None
        self.stepsize = None
        self.stepsize_base = None
        self.octaves = None
        self.octave_cutoff = None
        self.octave_scale = None
        self.iteration_mult = None
        self.step_mult = None
        self.jitter = 320
        self.package_name = None

        self.choose_model(data.program[self.current_program]['model'])
        #self.set_endlayer(self.layers[self.current_layer])
        #self.set_featuremap()
        self.cyclefx = None # contains cyclefx list for current program
        self.stepfx = None # contains stepfx list for current program


    def choose_model(self, key):
        self.net_fn = '{}/{}/{}'.format(self.models['path'], self.models[key][0], self.models[key][1])
        self.param_fn = '{}/{}/{}'.format(self.models['path'], self.models[key][0], self.models[key][2])
        self.caffemodel = self.models[key][2]

        # Patch model to be able to compute gradients
        # load the empty protobuf model
        model = caffe.io.caffe_pb2.NetParameter()

        # load the prototxt and place it in the empty model
        text_format.Merge(open(self.net_fn).read(), model)

        # add the force backward: true value
        model.force_backward = True

        # save it to a new file called tmp.prototxt
        open('tmp.prototxt', 'w').write(str(model))

        # the neural network model
        self.net = caffe.Classifier('tmp.prototxt',
            self.param_fn, mean=np.float32([104.0, 116.0, 122.0]), channel_swap=(2, 1, 0))

        update_HUD_log('model',self.caffemodel)

    def showlayers(self):
        # outputs valid layer list for this model
        print self.net.blobs.keys()

    def set_program(self, index):
        self.package_name = data.program[index]['name']
        self.iterations = data.program[index]['iterations']
        self.stepsize_base = data.program[index]['step_size']
        self.octaves = data.program[index]['octaves']
        self.octave_cutoff = data.program[index]['octave_cutoff']
        self.octave_scale = data.program[index]['octave_scale']
        self.iteration_mult = data.program[index]['iteration_mult']
        self.step_mult = data.program[index]['step_mult']
        self.layers = data.program[index]['layers']
        self.features = data.program[index]['features']
        self.current_feature = 0;
        self.model = data.program[index]['model']
        self.choose_model(self.model)
        self.set_endlayer(self.layers[0])
        self.set_featuremap()
        self.cyclefx = data.program[index]['cyclefx']
        self.stepfx = data.program[index]['stepfx']
        self.program_start_time = time.time()
        log.critical('program:{} started:{}'.format(self.program[self.current_program]['name'], self.program_start_time))


    def set_endlayer(self,end):
        self.end = end
        Viewport.force_refresh = True
        update_HUD_log('layer','{}/{}'.format(end,self.net.blobs[self.end].data.shape[1]))

    def prev_layer(self):
        self.current_layer -= 1
        if self.current_layer < 0:
            self.current_layer = len(self.layers)-1
        self.set_endlayer(self.layers[self.current_layer])

    def next_layer(self):
        self.current_layer += 1
        if self.current_layer > len(self.layers)-1:
            self.current_layer = 0
        self.set_endlayer(self.layers[self.current_layer])

    def set_featuremap(self):
        Viewport.force_refresh = True
        featuremap = self.features[self.current_feature]
        log.critical('featuremap:{}'.format(featuremap))
        update_HUD_log('featuremap',featuremap)

    def prev_feature(self):
        max_feature_index = self.net.blobs[self.end].data.shape[1]
        self.current_feature -= 1
        if self.current_feature < 0:
            self.current_feature = len(self.features)-1
        self.set_featuremap()

    def next_feature(self):
        max_feature_index = self.net.blobs[self.end].data.shape[1]
        self.current_feature += 1

        if self.current_feature > len(self.features)-1:
            self.current_feature = 0
        if self.current_feature > max_feature_index-1:
            self.current_feature = -1
        self.set_featuremap()

    def reset_feature(self):
        pass

    def prev_program(self):
        self.current_program -= 1
        if self.current_program < 0:
            self.current_program = len(self.program)-1
        self.set_program(self.current_program)

    def next_program(self):
        self.current_program += 1
        if self.current_program > len(self.program)-1:
            self.current_program = 0
        self.set_program(self.current_program)

class Viewport(object):

    def __init__(self, window_name, username, listener):
        self.window_name = window_name
        self.b_show_HUD = False
        self.keypress_mult = 3 # accelerate value changes when key held
        self.b_show_stats = False
        self.motiondetect_log_enabled = False
        self.blend_ratio = 0.0
        self.username = username
        self.imagesavepath = '/home/gary/Pictures/'+self.username
        self.listener = listener
        self.force_refresh = True
        self.image = None
        self.time_counter = 0
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def show(self, image):
        self.time_counter += 1
        # convert and clip floating point matrix into RGB bounds as integers
        image = np.uint8(np.clip(image, 0, 224)) # TODO valid practice still? tweaked to get some hilites in output

        # resize image to fit viewport, skip if already at full size
        if image.shape[0] != Display.height:
            image = cv2.resize(image, (Display.width, Display.height), interpolation = cv2.INTER_LINEAR)

        image = Composer.update(image)

        image = self.postfx(image) # HUD
        if self.b_show_stats:
            image = self.postfx2(image) # stats
        cv2.imshow(self.window_name, image)

        self.monitor()
        self.listener()

        # GRB: temp structure for saving fully rendered frames
        if self.time_counter > 10 and self.username != 'silent':
            self.export(image)
            self.time_counter = 0
        self.image = image

    def export(self,image):
        make_sure_path_exists(self.imagesavepath)
        log.debug('{}:{}'.format('export image',self.imagesavepath))
        export_path = '{}/{}.jpg'.format(
            self.imagesavepath,
            time.strftime('%m-%d-%H-%M-%s')
            )
        savefile = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        PIL.Image.fromarray(np.uint8(savefile)).save(export_path)
        #tweet(export_path)

    # forces new cycle with new camera image
    def refresh(self):
        self.force_refresh = True

    def postfx(self, image):

        # this would be a good place for color processing that
        # only affects the output (i.e. not cycled back into the net)
        # image = vignette(image,400)
        if self.b_show_HUD:
            image = show_HUD(image)
        return image

    def postfx2(self, image):
        image = show_stats(image)
        return image

    def monitor(self):
        if self.motiondetect_log_enabled:
            img = MotionDetector.t_delta_framebuffer

            # composite motion stats here
            # rectangle
            overlay = MotionDetector.t_delta_framebuffer.copy()
            opacity = 1.0
            #cv2.rectangle(overlay,(0,0),(Display.width, Display.height), (0, 0, 0), -1)
            cv2.putText(overlay, MotionDetector.monitor_msg, (30, Display.height - 100), FONT, 0.5, WHITE)
            # add overlay back to source
            img = cv2.addWeighted(overlay, opacity, img, 1-opacity, 0, img)
            cv2.imshow('delta', img)


    def shutdown(self):
        cv2.destroyAllWindows()
        for cam in Camera:
            cam.stop()
        sys.exit()

class Composer(object):
    # both self.buffer1 and self.buffer2 look to data.capture_size for their dimensions
    # this happens on init
    def __init__(self):
        self.is_dirty = False # the type of frame in buffer1. dirty when recycling clean when refreshing
        self.is_new_cycle = True
        self.buffer1 = Webcam.get().read() # uses camera capture dimensions
        self.buffer2 = Webcam.get().read() # uses camera capture dimensions
        self.opacity = 1.0
        self.is_compositing_enabled = False
        self.xform_scale = 0.03

    def update(self, image):
        if self.is_dirty:
            if self.is_new_cycle:
                for fx in Model.cyclefx:
                    if fx['name'] == 'inception_xform':
                        self.buffer1 = FX.inception_xform(image, Webcam.get().capture_size, **fx['params'])
                # Viewport.export(self.buffer1)

            self.is_dirty = False
            self.is_compositing_enabled = False

        else:
            if self.is_new_cycle and MotionDetector.isResting() == False:
                self.is_compositing_enabled = True

            if self.is_compositing_enabled:
                image = cv2.addWeighted(self.buffer2, self.opacity, image, 1-self.opacity, 0, image)
                self.opacity = self.opacity * 0.9
                if self.opacity <= 0.01:
                    self.opacity = 1.0
                    self.is_compositing_enabled = False

        return image

    def write_buffer2(self,image):
        if self.is_compositing_enabled == False:
            # convert and clip floating point matrix into RGB bounds
            self.buffer2 = np.uint8(np.clip(image, 0, 255))

            ### resize buffer 2 to match viewport dimensions
            if image.shape[1] != Display.width:
                self.buffer2 = cv2.resize(self.buffer2, (Display.width, Display.height), interpolation = cv2.INTER_LINEAR)
        return

class FX(object):
    def __init__(self):
        self.direction = 1
        self.stepfx_opacity = 1.0
        self.cycle_start_time = 0
        self.program_start_time = 0

    def xform_array(self, img, amplitude, wavelength):
        def shiftfunc(n):
            return int(amplitude*np.sin(n/wavelength))
        for n in range(img.shape[1]): # number of rows in the image
            img[:, n] = np.roll(img[:, n], 3*shiftfunc(n))
        return img

    def test_args(self, model=Model, step=0.05, min_scale=1.2, max_scale=1.6):
        print 'model: ', model
        print 'step: ', step
        print 'min_scale: ', min_scale
        print 'max_scale: ', max_scale

    def octave_scaler(self, model=Model, step=0.05, min_scale=1.2, max_scale=1.6):
        # octave scaling cycle each rem cycle, maybe
        # if (int(time.time()) % 2):
        model.octave_scale += step * self.direction
        # hackish, but prevents values from getting stuck above or beneath min/max
        if model.octave_scale > max_scale or model.octave_scale <= min_scale:
            self.direction = -1 * self.direction
        update_HUD_log('scale',model.octave_scale)
        log.info('octave_scale: {}'.format(model.octave_scale))

    def inception_xform(self, image, capture_size, scale):
        # return nd.affine_transform(image, [1-scale, 1, 1], [capture_size[1]*scale/2, 0, 0], order=1)
        return nd.affine_transform(image, [1-scale, 1-scale, 1], [capture_size[0]*scale/2, capture_size[1]*scale/2, 0], order=1)

    def median_blur(self, image, kernel_shape):
        return cv2.medianBlur(image, kernel_shape)

    def bilateral_filter(self, image, radius, sigma_color, sigma_xy):
        return cv2.bilateralFilter(image, radius, sigma_color, sigma_xy)

    def nd_gaussian(self, image, sigma, order):
        return nd.filters.gaussian_filter(image, sigma, order)

    def step_mixer(self,opacity):
        self.stepfx_opacity = opacity

    def duration_cutoff(self, duration):
        elapsed = time.time() - self.cycle_start_time
        if elapsed >= duration:
            MotionDetector.force_detection()
            Viewport.refresh()
        log.debug('cycle_start_time:{} duration:{} elapsed:{}'.format(self.cycle_start_time, duration, elapsed))

    # called by main() at start of each cycle
    def set_cycle_start_time(self, start_time):
        self.cycle_start_time = start_time


# stepfx wrapper takes neural net data blob and converts to caffe image
# then after processing, takes the caffe image and converts back to caffe
def iterationPostProcess(net, net_data_blob):
    img = caffe2rgb(net, net_data_blob)
    img2 = img.copy()

    #  apply stepfx, assuming they've been defined
    if Model.stepfx is not None:
        for fx in Model.stepfx:
            if fx['name'] == 'median_blur':
                img2 = FX.median_blur(img2, **fx['params'])

            if fx['name'] == 'bilateral_filter':
                img2 = FX.bilateral_filter(img2, **fx['params'])

            if fx['name'] == 'nd_gaussian':
                img2 = FX.nd_gaussian(img2, **fx['params'])

            if fx['name'] == 'step_opacity':
                FX.step_mixer(**fx['params'])

            if fx['name'] == 'duration_cutoff':
                FX.duration_cutoff(**fx['params'])

    img = cv2.addWeighted(img2, FX.stepfx_opacity, img, 1.0-FX.stepfx_opacity, 0, img)
    return rgb2caffe(Model.net, img)


def blur(img, sigmax, sigmay):
    img2 = img.copy()
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # if (int(time.time()) % 0.5):
    #     return img

    # img = cv2.medianBlur(img,sigmax)

    # A
    # img = cv2.bilateralFilter(img, 17, 30, 3)
    # img = cv2.bilateralFilter(img, 10, 30, 30)

    #B
    # img = cv2.bilateralFilter(img, 7, 50, 3)
    # img = cv2.medianBlur(img,3)

    #C
    # extreme grey cartoon
    # img = nd.filters.gaussian_filter(img, 0.6, order=0)
    # img = cv2.bilateralFilter(img, 13, 20, 120)

    # eurogoth (work-in-progress)
    # img = nd.filters.gaussian_filter(img, 0.7, order=0)
    # img = cv2.bilateralFilter(img, 3, 70, 50)

    #D monoworld
    # img2 = nd.filters.gaussian_filter(img2, 0.5, order=0)
    # img2 = cv2.medianBlur(img2,3)

    # simplified
    # works best with high iteration counts
    # img = cv2.medianBlur(img,5)

    # return img2
    return cv2.addWeighted(img2, FX.stepfx_opacity, img, 1.0-FX.stepfx_opacity, 0, img)


def vignette(img,param):
    rows,cols = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols,param)
    kernel_y = cv2.getGaussianKernel(rows,param)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    output = np.copy(img)
    for i in range(3):
        output[:,:,i] = np.uint8(np.clip((output[:,:,i] * mask ), 0, 2))
    return output

def sobel(img):
    xgrad = nd.filters.sobel(img, 0)
    ygrad = nd.filters.sobel(img, 1)
    combined = np.hypot(xgrad, ygrad)
    sob = 255 * combined / np.max(combined) # normalize
    log.debug('@@@@@@@@@@ sobel{}'.format(sobel.shape))
    return sob


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

    #myStatusText = '{} #deepdreamvisionquest #gdc2016'.format(Viewport.username)
    myStatusText = '#deepdreamvisionquest #gdc2016 test'
    api.update_with_media(path_to_image, status=myStatusText )

def update_HUD_log(key,new_value):
    hud_log[key][1] = hud_log[key][0]
    hud_log[key][0] = new_value


def show_stats(image):
    stats_overlay = image.copy()
    opacity = 0.9
    cv2.putText(stats_overlay, 'show_stats()', (30, 40), font, 0.5, GREEN)
    return cv2.addWeighted(stats_overlay, opacity, image, 1-opacity, 0, image)


def show_HUD(image):
    # rectangle
    overlay = image.copy()
    opacity = 0.5
    cv2.rectangle(overlay,(0,0),(Display.width, Display.height), (0, 0, 0), -1)
    #cv2.rectangle(image_to_draw_on, (x1,y1), (x2,y2), (r,g,b), line_width )

    # list setup
    x,xoff = 40,180
    y,yoff = 150,35

    data.counter = 0
    def write_Text(key):
        color = WHITE
        row = y + yoff * data.counter
        if hud_log[key][0] != hud_log[key][1]:
            #  value has changed since last update
            color = GREEN
            hud_log[key][1] = hud_log[key][0] #  update history
        cv2.putText(overlay, key, (x, row), FONT, 0.5, WHITE)
        cv2.putText(overlay, '{}'.format(hud_log[key][0]), (xoff, row), FONT, 1.0, color)

        data.counter += 1

    # write text to overlay
    # col1
    cv2.putText(overlay, hud_log['detect'][0], (x, 40), FONT, 1.0, (0,255,0))
    cv2.putText(overlay, 'DEEPDREAMVISIONQUEST', (x, 100), FONT, 0.5, WHITE)
    write_Text('program')
    write_Text('floor')
    write_Text('threshold')
    write_Text('last')
    write_Text('now')
    write_Text('model')
    write_Text('layer')
    write_Text('featuremap')
    write_Text('width')
    write_Text('height')
    write_Text('scale')
    write_Text('octave')
    write_Text('iteration')
    write_Text('step_size')
    write_Text('cycle_time')


    # add overlay back to source
    return cv2.addWeighted(overlay, opacity, image, 1-opacity, 0, image)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

# keyboard event handler
def listener():
    key = cv2.waitKey(1) & 0xFF
    # log.critical('key pressed: {}'.format(key))

    # Row A
    # --------------------------------

    if key==10: # ENTER key: save picture
        log.critical('{}:{} {} {}'.format('A1',key,'ENTER','SAVE IMAGE'))
        Viewport.export()
        return

    if key==32: # SPACE
        log.critical('{}:{} {} {}'.format('A2',key,'SPACE','***'))

    if key==80: # HOME
        log.critical('{}:{} {} {}'.format('A3',key,'HOME','***'))
        return

    if key==87: # HOME
        log.critical('{}:{} {} {}'.format('A4',key,'END','RESET'))
        return

    # Row B
    # --------------------------------

    if key==85: # PAGE UP: Previous Bank
        log.critical('{}:{} {} {}'.format('B1',key,'PAGEUP','BANK-'))
        return

    if key==86: # PAGE DOWN: NEXT Bank
        log.critical('{}:{} {} {}'.format('B2',key,'PAGEDOWN','BANK+'))
        return

    if key == 81: # left-arrow key: previous program
        log.critical('x{}:{} {} {}'.format('B3',key,'ARROWL','PROGRAM-'))
        Model.prev_program()
        Model.reset_feature()
        return

    if key == 83: # right-arrow key: next program
        log.critical('{}:{} {} {}'.format('B4',key,'ARROWR','PROGRAM+'))
        Model.next_program()
        Model.reset_feature()
        return

    # Row C
    # --------------------------------
    if key==194: # F5
        log.critical('{}:{} {} {}'.format('C1',key,'F5','***'))
        return

    if key==195: # F6
        log.critical('{}:{} {} {}'.format('C2',key,'F6','***'))
        return

    if key == 122: # z key: next network layer
        log.critical('{}:{} {} {}'.format('C3',key,'Z','LAYER-'))
        Model.prev_layer()
        return

    if key == 120: # x key: previous network layer
        log.critical('{}:{} {} {}'.format('C4',key,'X','LAYER+'))
        Model.next_layer()
        return

    # Row D
    # --------------------------------

    elif key==196: # F7
        log.critical('{}:{} {} {}'.format('D1',key,'F7','***'))

    elif key==197: # F8
        log.critical('{}:{} {} {}'.format('D2',key,'F8','***'))


    elif key == 44: # , key : previous featuremap
        log.critical('{}:{} {} {}'.format('D3',key,',','Feature-'))
        Model.prev_feature()

    elif key == 46: # . key : next featuremap
        log.critical('{}:{} {} {}'.format('D4',key,'.','Feature+'))
        Model.next_feature()

    # Row E
    # --------------------------------

    if key==91: # [
        log.critical('{}:{} {} {}'.format('E1',key,'[','***'))

    if key==93: # ]
        log.critical('{}:{} {} {}'.format('E2',key,']','***'))

    if key == 45: # _ key (underscore) : decrease motion threshold
        MotionDetector.floor -= 1000 * Viewport.keypress_mult
        if MotionDetector.floor < 1:
            MotionDetector.floor = 1
        update_HUD_log('floor',MotionDetector.floor)
        log.critical('{}:{} {} {}'.format('E3',key,'-','FLOOR-'))
        return

    if key == 61: # = key (equals): increase motion threshold
        MotionDetector.floor += 1000 * Viewport.keypress_mult
        update_HUD_log('floor',MotionDetector.floor)
        log.critical('{}:{} {} {}'.format('E4',key,'=','FLOOR+'))

        #temp
        Model.showlayers()
        return

    # Row F
    # --------------------------------

    # dssabled for single camera show
    # if key == 190: # F1 key: Toggle Camera
    #     index = (Webcam.current + 1) % 2 # hardcoded for 2 cameras
    #     MotionDetector.camera = Webcam.set(Device[index])
    #     log.critical('{}:{} {} {}'.format('F1',key,'F1','TOGGLE CAMERA'))
    #     return

    if key == 112: # p key : pause/unpause motion detection
        MotionDetector.is_paused = not MotionDetector.is_paused
        if not MotionDetector.is_paused:
            MotionDetector.delta_trigger = MotionDetector.delta_trigger_history
        log.critical('{}:{} {} {}'.format('F2',key,'P','TOGGLE MOTION'))
        return

    if key == 96: # `(tilde) key: toggle HUD
        Viewport.b_show_HUD = not Viewport.b_show_HUD
        log.critical('{}:{} {} {}'.format('F3',key,'`','TOGGLE HUD'))
        return

    if key == 49: # 1 key : toggle motion detect window
        Viewport.motiondetect_log_enabled = not Viewport.motiondetect_log_enabled
        if Viewport.motiondetect_log_enabled:
            cv2.namedWindow('delta',cv2.WINDOW_AUTOSIZE)
        else:
            cv2.destroyWindow('delta')
        log.critical('{}:{} {} {}'.format('F4',key,'1','MOTION MONITOR'))
        return

    # --------------------------------

    if key == 27: # ESC: Exit
        log.critical('{}:{} {} {}'.format('**',key,'ESC','SHUTDOWN'))
        Viewport.shutdown()

        # close the motion detector data export file
        MotionDetector.export.close()
        return


# a couple of utility functions for converting to and from Caffe's input image layout
def rgb2caffe(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

def caffe2rgb(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def objective_L2(dst):
    dst.diff[:] = dst.data

def objective_guide(dst):
    x = dst.data[0].copy()
    y = Model.guide_features
    ch = x.shape[0]
    x = x.reshape(ch,-1)
    y = y.reshape(ch,-1)
    A = x.T.dot(y) # compute the matrix of dot produts with guide features
    dst.diff[0].reshape(ch,-1)[:] = y[:,A.argmax(1)] # select one that matches best


# -------
# implements forward and backward passes thru the network
# apply normalized ascent step upon the image in the networks data blob
# supports Feature Map activation
# -------
def make_step(net, step_size=1.5, end='inception_4c/output', jitter=32, clip=True, feature=-1):

    log.info('step_size:{} feature:{} end:{}\n{}'.format(step_size, feature, end,'-'*10))
    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2)
    net.forward(end=end)

    # feature inspection
    if feature == -1:
        dst.diff[:] = dst.data
    else:
        dst.diff.fill(0.0)
        dst.diff[0,feature,:] = dst.data[0,feature,:]

    net.backward(start=end)
    g = src.diff[0]

    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * (g*step_size)

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 200-bias)

    # postprocessor
    src.data[0] = iterationPostProcess(net, src.data[0])

    # sequencer
    program_elapsed_time = time.time() - Model.program_start_time
    if program_elapsed_time > Model.program_duration:
        # if Model.program_bank
        Model.next_program()


# -------
# REM CYCLE
# -------
def deepdream(net, base_img, iteration_max=10, octave_n=4, octave_scale=1.4, end='inception_4c/output', **step_params):
    # COOLDOWN
    # returns the camera on the first update of a new cycle 
    # after previously being kicked out of deepdream
    # effectively keeps the Composer in sync
        # disabling this prevents the previous camera capture from being flushed
        # (we end up seeing it as a ghost image before hallucination begins on the new camera)


    # GRB: Not entirely sure why this condition gets triggered 
    # noticing it when the system starts up. does it appear at other times? when?
    if MotionDetector.wasMotionDetected:
        Composer.write_buffer2(Webcam.get().read())
        Composer.is_dirty = False # no, we'll be refreshing the frane buffer
        return Webcam.get().read()

    


    # SETUPOCTAVES---
    Composer.is_new_cycle = False
    src = Model.net.blobs['data']
    octaves = [rgb2caffe(Model.net, base_img)]
    for i in xrange(octave_n - 1):
        octaves.append(nd.zoom(octaves[-1], (1, round((1.0 / octave_scale),2), round((1.0 / octave_scale),2)), order=1))
    detail = np.zeros_like(octaves[-1])

    # OCTAVEARRAY CYCLE, last (smallest) octave first
    for octave, octave_current in enumerate(octaves[::-1]):
        h, w = octave_current.shape[-2:]
        h1, w1 = detail.shape[-2:]
        detail = nd.zoom(detail, (1, 1.0 * h / h1, 1.0 * w / w1), order=0)

        # reshape the network's input image data to match octave_current shape
        src.reshape(1,3,h,w)

        # add the changed details to the network's input image
        src.data[0] = octave_current + detail

        Model.stepsize = Model.stepsize_base # reset step size to default each octave
        step_params['step_size'] = Model.stepsize # modifying the **step_params list for makestep

        # OCTAVECYCLE
        i=0
        while i < iteration_max:
            # FORCE REFRESH
            if Viewport.force_refresh:
                Composer.write_buffer2(Webcam.get().read())
                Composer.is_dirty = False # no, we'll be refreshing the frane buffer
                return Webcam.get().read()

            MotionDetector.process()
            if not MotionDetector.isResting():
                break

            # delegate gradient ascent to step function
            log.info('{:02d} {:02d} {:02d}'.format(octave,i,iteration_max))
            make_step(Model.net, end=end, **step_params)

            # write netblob to Composer
            Composer.buffer1 = caffe2rgb(Model.net, src.data[0])
            Viewport.show(Composer.buffer1)

            # attenuate step size over rem cycle
            x = step_params['step_size']
            step_params['step_size'] += x * Model.step_mult * 1.0

            # set a floor for any cyuclefx step modification
            if step_params['step_size'] < 1.1:
                step_params['step_size'] = 1.1

            i += 1

            # logging
            octavemsg = '{}/{}({})'.format(octave,octave_n,Model.octave_cutoff)
            guidemsg = '({}/{}) {}'.format(Model.current_guide,len(Model.guides),Model.guides[Model.current_guide])
            iterationmsg = '{:0>3}:{:0>3} x{}'.format(i,iteration_max,Model.iteration_mult)
            stepsizemsg = '{:02.3f} x{:02.3f}'.format(step_params['step_size'],Model.step_mult)
            thresholdmsg = '{:0>6}'.format(MotionDetector.delta_trigger)
            floormsg = '{:0>6}'.format(MotionDetector.floor)
            update_HUD_log('octave',octavemsg)
            update_HUD_log('width',w)
            update_HUD_log('height',h)
            update_HUD_log('guide',guidemsg)
            update_HUD_log('iteration',iterationmsg)
            update_HUD_log('step_size',stepsizemsg)
            update_HUD_log('scale',Model.octave_scale)
            update_HUD_log('program',Model.package_name)
            update_HUD_log('threshold',thresholdmsg)
            update_HUD_log('floor',floormsg)


        # probably temp? export each completed iteration
        # Viewport.export(Composer.buffer1)



        # CUTOFF
        # this turned out to be the last octave calculated in the series
        if octave == Model.octave_cutoff:
            Composer.is_dirty = True
            return caffe2rgb(Model.net, src.data[0])

        # EARLY EXIT
        # motion detected so we're ending this REM cycle
        if MotionDetector.wasMotionDetected:
            Composer.write_buffer2(caffe2rgb(Model.net, src.data[0]))
            Composer.is_dirty = False # no, we'll be refreshing the frane buffer
            return Webcam.get().read()

        # extract details produced on the current octave
        detail = src.data[0] - (octave_current-(randint(0,9)-5))

        # reduce iteration count for the next octave
        iteration_max = int(iteration_max - (iteration_max * Model.iteration_mult))
        #iteration_max = Model.next_iteration(iteration_max)

    # return the resulting image (converted back to x,y,RGB structured matrix)
    Composer.is_dirty = True # yes, we'll be recycling the Composer

    # return rendered img
    return caffe2rgb(Model.net, src.data[0])


# -------
# MAIN
# ------- 
def main():

    # start timer
    now = time.time()

    # set GPU mode
    caffe.set_device(0)
    caffe.set_mode_gpu()

    # parameters
    #Model.set_program('hirez-fast')
    iterations = Model.iterations
    stepsize = Model.stepsize_base
    octaves = Model.octaves
    octave_scale = Model.octave_scale
    jitter = 300
    update_HUD_log('model',Model.caffemodel)
    update_HUD_log('username',Viewport.username)
    update_HUD_log('settings',Model.package_name)


    # the madness begins
    Composer.buffer1 = Webcam.get().read() # initial camera image for init

    while True:
        log.info('new cycle')
        FX.set_cycle_start_time(time.time()) # register cycle start for duration_cutoff stepfx
        Composer.is_new_cycle = True
        # Viewport.export(Composer.buffer1)
        Viewport.show(Composer.buffer1)
        MotionDetector.process()
        if MotionDetector.wasMotionDetected:
            Composer.is_dirty = False

        if Composer.is_dirty == False or Viewport.force_refresh:
            # Viewport.save_next_frame = True

            #  apply cyclefx, assuming they've been defined
            if Model.cyclefx is not None:
                for fx in Model.cyclefx:
                    if fx['name'] == 'xform_array':
                        FX.xform_array(Composer.buffer1, **fx['params'])

                    if fx['name'] == 'octave_scaler':
                        FX.octave_scaler(model=Model, **fx['params'])

            # kicks off rem sleep - will begin continual iteration of the image through the model
            Composer.buffer1 = deepdream(net, Composer.buffer1, iteration_max = Model.iterations, octave_n = Model.octaves, octave_scale = Model.octave_scale, step_size = Model.stepsize_base, end = Model.end, feature = Model.features[Model.current_feature])

            # Viewport.export(Composer.buffer1)

            # commenting out this block allows unfiltered signal only
            if Viewport.force_refresh:
                Viewport.force_refresh = False


        # a bit later
        later = time.time()
        difference = later - now
        duration_msg = '{:.2f}s'.format(difference)
        update_HUD_log('cycle_time',duration_msg) # HUD
        log.info('cycle time: {}\n{}'.format(duration_msg,'-'*80))
        now = time.time() # the new now


# --------
# INIT
# --------

# setup system logging facilities
logging.config.dictConfig(LogSettings.LOGGING_CONFIG)
log = logging.getLogger('logtest-debug')
log.setLevel(logging.CRITICAL) # not sure why this works instead of definitions for this elsewhere


# log.debug('debug message!')
# log.info('info message!')
# log.error('error message')
# log.warning('warning message')
# log.critical('critical message')

hud_log = {
    # (current value, old value)
    'octave': [None,None],
    'width': [None,None],
    'height': [None,None],
    'guide': [None,None],
    'layer': [None,None],
    'last': [None,None],
    'now': [None,None],
    'iteration': [None,None],
    'step_size': [None,None],
    'settings': [None,None],
    'threshold': [None,None],
    'detect': [None,None],
    'cycle_time': [None,None],
    'featuremap': [None,None],
    'model': [None,None],
    'username': [None,None],
    'scale': [None,None],
    'program': [None,None],
    'floor': [None,None]
}

# HUD
# dictionary contains the key/values we'll be logging
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GREEN = (0,255,0)

# global reference to the neural network object
net = None


# camera setup
Camera = []

# note that camera index changes if a webcam is unplugged
# default values are  [0,1] for 2 camera setup
Device = [0,1] # debug

w = data.capture_w
h = data.capture_h
# single camera setuo will use camera index 0
Camera.append(WebcamVideoStream(Device[0], w, h, portrait_alignment=True, flip_h=False, flip_v=False, gamma=0.2).start())

# temp disable cam 2 for show setup
# Camera.append(WebcamVideoStream(Device[1], w, h, portrait_alignment=True, flip_h=False, flip_v=True, gamma=0.8).start())

Webcam = Cameras(source=Camera, current=Device[0])
Display = Display(width=w, height=h, camera=Webcam.get())

# need to set the floor value to reflect the amount of light in the area
MotionDetector = MotionDetector(floor=100000, camera=Webcam.get(), log=update_HUD_log)

# disable screen export when usename specified is 'silent'
Viewport = Viewport('deepdreamvisionquest','dev-1', listener)
Composer = Composer()
Model = Model(program_duration=60) # seconds
FX = FX()

Model.set_program(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--username',help='twitter userid for sharing')
    args = parser.parse_args()
    if args.username:
        Viewport.username = '@{}'.format(args.username)
    main()