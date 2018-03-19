# rem.py

# file handling
import os
import os.path
import argparse
import sys
import errno

# image processing
import scipy.ndimage as nd
import PIL.Image
import cv2

# math
import math
import numpy as np
from random import randint

# communication
import tweepy # was used to post the framebuffer to twitter

# neural network stuff
os.environ['GLOG_minloglevel'] = '2' # suppress verbose caffe logging before caffe import
import caffe
from google.protobuf import text_format

''' apparently unused '''
# import inspect
# from collections import deque # no idea. still valid?

# logging
import logging
import logging.config
sys.path.append('../bin') #  point to directory containing LogSettings
import LogSettings # global log settings templ

# system
from threading import Thread
import time

# program modules
import data
from camerautils import WebcamVideoStream
from camerautils import Cameras


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

        self.screensize = [self.width, self.height]


class Model(object):
    def __init__(self, current_layer=0, program_duration=-1):
        self.net = None
        self.net_fn = None
        self.param_fn = None
        self.caffemodel = None
        self.end = None
        self.models = data.models
        self.guides = data.guides
        self.guide_features = self.guides[0]

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
        self.installation_startup = time.time() # keep track of runtime

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

    def show_network_details(self):
        # outputs layer details to console
        print self.net.blobs.keys()
        print 'current layer:{} ({}) current feature:{}'.format(
            self.end,
            self.net.blobs[self.end].data.shape[1],
            self.features[self.current_feature]
            )

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
        log.warning('program:{} started:{}'.format(self.program[self.current_program]['name'], self.program_start_time))


    def set_endlayer(self,end):
        self.end = end
        Viewport.force_refresh = True
        log.warning('layer: {} ({})'.format(self.end,self.net.blobs[self.end].data.shape[1]))
        update_HUD_log('layer','{} ({})'.format(self.end,self.net.blobs[self.end].data.shape[1]))

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
        # featuremap = self.features[self.current_feature]
        log.warning('featuremap:{}'.format(self.features[self.current_feature]))
        update_HUD_log('featuremap',self.features[self.current_feature])

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
        self.window_name = '{}/{}'.format(window_name, username)
        self.username = username
        self.b_show_HUD = False
        self.keypress_mult = 3 # accelerate value changes when key held
        self.b_show_stats = True
        self.motiondetect_log_enabled = False
        self.blend_ratio = 0.0
        self.imagesavepath = '/home/gary/Pictures/'+self.username
        self.listener = listener
        self.force_refresh = True
        self.image = None
        self.time_counter = 0
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def show(self, image):
        self.time_counter += 1
        image = np.uint8(np.clip(image, 0, 255)) # cast floating point matrix to RGB bounds as int
        image = Composer.update(image) # post process with cycleFX

        if self.b_show_HUD: # HUD overlay
            image = draw_HUD(image)
        cv2.imshow(self.window_name, image) # draw to window

        self.monitor() # handle motion detection viewport
        self.listener() # listen for keyboard events

        # GRB: temp structure for saving fully rendered frames
        if self.time_counter > 1 and self.username != 'silent':
            self.export(image)
            self.time_counter = 0
        self.image = image

    def export(self,image):
        make_sure_path_exists(self.imagesavepath)
        log.warning('{}:{}'.format('export image',self.imagesavepath))
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

    def monitor(self):
        if self.motiondetect_log_enabled:
            img = Webcam.get().t_delta_framebuffer # pointer to the motion detectors framebuffer


            resizedimg = img.copy()

            # ### resize channel to match viewport dimensions
            # if img.shape[1] != Display.width:
            #     resizedimg = cv2.resize(resizedimg, (Display.width, Display.height), interpolation = cv2.INTER_LINEAR)

            # composite motion stats here
            overlay = img.copy()
            opacity = 1.0
            cv2.putText(overlay,Webcam.get().motiondetector.monitor_msg, (20, 20), FONT, 0.5, WHITE)
            img = cv2.addWeighted(overlay, opacity, img, 1-opacity, 0, img) # add overlay back to source
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
        self.isDreaming = False
        self.opacity = 1.0
        self.is_compositing_enabled = False
        self.xform_scale = 0.03
        self.buffer = []
        self.buffer.append( Webcam.get().read() ) # uses camera capture dimensions
        self.buffer.append( Webcam.get().read() ) # uses camera capture dimensions
        # self.buffer1 = Webcam.get().read() # uses camera capture dimensions
        # self.buffer2 = Webcam.get().read() # uses camera capture dimensions
        self.mixbuffer = np.zeros((Display.height, Display.width ,3), np.uint8)
        self.dreambuffer = Webcam.get().read() # uses camera capture dimensions

        #
        self.mix_opacity = 0.0 # the mix between dreaming and webcam buffers, 1.0 = all webcam
        self.ramp_stopped = False
        self.ramp_toggle_flag = False
        self.ramp_counter = 0
        self.ramp_increment = 0
        self.b_cycle = False

        # maybe ?
        self.force_refresh = True

    def send(self, channel, img):
         # route input img to channel

        self.buffer[channel] = img
        log.critical('#324 self.buffer[{}] shape: {}'.format(channel, self.buffer[channel].shape))

        ### resize channel to match viewport dimensions
        if img.shape[1] != Display.width:
            self.buffer[channel] = cv2.resize(self.buffer[channel], (Display.width, Display.height), interpolation = cv2.INTER_LINEAR)

        # convert and clip any floating point values into RGB bounds as integers
        self.buffer[channel] = np.uint8(np.clip(self.buffer[channel], 0, 255))

    def mix(self, img_back, img_front, mix_opacity):
        self.mix_opacity = self.ramp_counter
        cv2.addWeighted(
            img_front,
            self.mix_opacity,#
            img_back,
            1-self.mix_opacity,
            0,
            self.mixbuffer
            )

        return self.mixbuffer

    def update(self, image):
        if self.isDreaming:
            for fx in Model.cyclefx:
                if fx['name'] == 'inception_xform':
                    image = FX.inception_xform(image, Display.screensize, **fx['params'])
                    Composer.dreambuffer = image

            self.isDreaming = False

        return image

    def ramp_start(self):
        Thread(target=self.ramp_update, args=()).start()
        log.debug('ramp start')
        return self

    def ramp_update(self):
        # loop until the thread is stopped

        while True:
            if self.ramp_stopped:
                log.debug('ramp stopped')
                return

            if self.ramp_toggle_flag:
                # self.ramp_counter = 1.0
                # self.b_cycle = True

                # if self.b_cycle:
                self.ramp_increment = -0.1
                time.sleep(0.2)

                if self.ramp_counter <= 0.1:
                    self.ramp_toggle_flag = False
            else:
                self.ramp_increment = 0
                self.ramp_counter = 0.0
                self.b_cycle = False

            self.ramp_counter += self.ramp_increment

        return

    def ramp_toggle(self, b_state=True):
        self.ramp_toggle_flag = b_state
        self.ramp_counter = 1.0


    def ramp_stop(self):
        log.debug('ramp stop')
        self.ramp_stopped = True



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
        # prevents values from getting stuck above or beneath min/max
        if model.octave_scale > max_scale or model.octave_scale <= min_scale:
            self.direction = -1 * self.direction
        update_HUD_log('scale',model.octave_scale)
        log.warning('octave_scale: {}'.format(model.octave_scale))

    def inception_xform(self, image, capture_size, scale):
        return nd.affine_transform(image, [1-scale, 1, 1], [capture_size[1]*scale/2, 0, 0], order=1)
        # return nd.affine_transform(image, [1-scale, 1-scale, 1], [capture_size[0]*scale/2, capture_size[1]*scale/2, 0], order=1)

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
            Webcam.get().motiondetector.force_detection()
            Viewport.refresh()
        log.warning('cycle_start_time:{} duration:{} elapsed:{}'.format(self.cycle_start_time, duration, elapsed))

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


# rename this function plz
def blur(img, sigmax, sigmay):
    img2 = img.copy()
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
    log.warning('show stats')
    stats_overlay = image.copy()
    opacity = 1.0
    cv2.putText(stats_overlay, 'show_stats()', (30, 40), font, 0.5, RED)
    return cv2.addWeighted(stats_overlay, opacity, image, 1-opacity, 0, image)


def draw_HUD(image):
    # rectangle
    overlay = image.copy()
    opacity = 0.5
    cv2.rectangle(overlay,(0,0),(Display.width, Display.height), (0, 0, 0), -1)
    #cv2.rectangle(image_to_draw_on, (x1,y1), (x2,y2), (r,g,b), line_width )

    # list setup
    x,xoff = 40,180
    y,yoff = 150,20

    data.counter = 0
    def write_Text(key):
        color = WHITE
        row = y + yoff * data.counter
        if hud_log[key][0] != hud_log[key][1]:
            #  value has changed since last update
            color = GREEN
            hud_log[key][1] = hud_log[key][0] #  update history
        cv2.putText(overlay, key, (x, row), FONT, 0.5, WHITE)
        cv2.putText(overlay, '{}'.format(hud_log[key][0]), (xoff, row), FONT, 0.5, color)

        data.counter += 1

    # write text to overlay
    # col1
    cv2.putText(overlay, hud_log['detect'][0], (x, 40), FONT, 1.0, (0,255,0))
    cv2.putText(overlay, 'DEEPDREAMVISIONQUEST', (x, 100), FONT, 1.0, WHITE)
    write_Text('program')
    write_Text('interval')
    write_Text('runtime')
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
    write_Text('gamma')


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
        log.warning('{}:{} {} {}'.format('A1',key,'ENTER','SAVE IMAGE'))
        Viewport.export()
        return

    if key==32: # SPACE
        log.warning('{}:{} {} {}'.format('A2',key,'SPACE','***'))
        return

    if key==80: # HOME
        log.warning('{}:{} {} {}'.format('A3',key,'HOME','***'))
        return

    if key==87: # END
        log.warning('{}:{} {} {}'.format('A4',key,'END','RESET'))
        return

    # Row B
    # --------------------------------

    if key==85: # PAGE UP : increase gamma
        log.warning('{}:{} {} {}'.format('B1',key,'PAGEUP','GAMMA+'))
        Webcam.get().gamma += 0.1
        Webcam.get().update_gamma(Webcam.get().gamma )
        return

    if key==86: # PAGE DOWN decrease gamma
        log.warning('{}:{} {} {}'.format('B2',key,'PAGEDOWN','GAMMA-'))
        Webcam.get().gamma -= 0.1
        Webcam.get().update_gamma(Webcam.get().gamma )
        return

    if key == 81: # left-arrow key: previous program
        log.warning('x{}:{} {} {}'.format('B3',key,'ARROWL','PROGRAM-'))
        Model.prev_program()
        Model.reset_feature()
        return

    if key == 83: # right-arrow key: next program
        log.warning('{}:{} {} {}'.format('B4',key,'ARROWR','PROGRAM+'))
        Model.next_program()
        Model.reset_feature()
        return

    # Row C
    # --------------------------------
    if key==194: # F5
        log.warning('{}:{} {} {}'.format('C1',key,'F5','***'))
        return

    if key==195: # F6
        log.warning('{}:{} {} {}'.format('C2',key,'F6','***'))
        return

    if key == 122: # z key: next network layer
        log.warning('{}:{} {} {}'.format('C3',key,'Z','LAYER-'))
        Model.prev_layer()
        Model.reset_feature()
        return

    if key == 120: # x key: previous network layer
        log.warning('{}:{} {} {}'.format('C4',key,'X','LAYER+'))
        Model.next_layer()
        Model.reset_feature()
        return

    # Row D
    # --------------------------------

    elif key==196: # F7: show network details
        log.warning('{}:{} {} {}'.format('D1',key,'F7','***'))
        # Model.show_network_details()
        Composer.ramp_toggle(True)
        return

    elif key==197: # F8
        log.warning('{}:{} {} {}'.format('D2',key,'F8','***'))
        Composer.ramp_toggle(False)
        return

    elif key == 44: # , key : previous featuremap
        log.warning('{}:{} {} {}'.format('D3',key,',','Feature-'))
        Model.prev_feature()

    elif key == 46: # . key : next featuremap
        log.warning('{}:{} {} {}'.format('D4',key,'.','Feature+'))
        Model.next_feature()

    # Row E
    # --------------------------------

    if key==91: # [
        log.warning('{}:{} {} {}'.format('E1',key,'[','***'))

    if key==93: # ]
        log.warning('{}:{} {} {}'.format('E2',key,']','***'))

    if key == 45: # _ key (underscore) : decrease detection floor
        Webcam.get().motiondetector.floor -= 100
        if Webcam.get().motiondetector.floor < 0:
            Webcam.get().motiondetector.floor = 0
        update_HUD_log('floor',Webcam.get().motiondetector.floor)
        log.warning('{}:{} {} {} :{}'.format('E3',key,'-','FLOOR-',Webcam.get().motiondetector.floor))
        return

    if key == 61: # = key (equals): increase detection floor
        Webcam.get().motiondetector.floor += 100
        update_HUD_log('floor',Webcam.get().motiondetector.floor)
        log.warning('{}:{} {} {} :{}'.format('E4',key,'=','FLOOR+',Webcam.get().motiondetector.floor))
        return

    # Row F
    # --------------------------------

    # dssabled for single camera show
    # if key == 190: # F1 key: Toggle Camera
    #     index = (Webcam.current + 1) % 2 # hardcoded for 2 cameras
    #     Webcam.get().motiondetector.camera = Webcam.set(Device[index])
    #     log.warning('{}:{} {} {}'.format('F1',key,'F1','TOGGLE CAMERA'))
    #     return

    if key == 112: # p key : pause/unpause motion detection
        Webcam.get().motiondetector.is_paused = not Webcam.get().motiondetector.is_paused
        log.warning('{}:{} {} {}:{}'.format('F2',key,'P','PAUSE',Webcam.get().motiondetector.is_paused))
        return

    if key == 96: # `(tilde) key: toggle HUD
        Viewport.b_show_HUD = not Viewport.b_show_HUD
        log.warning('{}:{} {} {}'.format('F3',key,'`','TOGGLE HUD'))
        return

    if key == 49: # 1 key : toggle motion detect window
        Viewport.motiondetect_log_enabled = not Viewport.motiondetect_log_enabled
        if Viewport.motiondetect_log_enabled:
            cv2.namedWindow('delta',cv2.WINDOW_AUTOSIZE)
        else:
            cv2.destroyWindow('delta')
        log.warning('{}:{} {} {}'.format('F4',key,'1','MOTION MONITOR'))
        return

    # --------------------------------

    if key == 27: # ESC: Exit
        ### the order of these operations is unreasonably specific
        log.warning('{}:{} {} {}'.format('**',key,'ESC','SHUTDOWN'))
        Composer.ramp_stop() # shutdown down the Composer ramp counter
        Viewport.shutdown()
        Webcam.get().motiondetector.export.close() # close the motion detector data export file
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
def make_step(net, step_size=1.5, end='inception_4c/output', jitter=500, clip=True, feature=-1, objective=objective_L2):

    log.info('step_size:{} feature:{} end:{}\n{}'.format(step_size, feature, end,'-'*10))
    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # shift image (jitter)
    net.forward(end=end)

    # feature inspection
    if feature == -1:
        objective(dst)
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
        src.data[:] = np.clip(src.data, -bias, 240-bias)

    # postprocessor
    src.data[0] = iterationPostProcess(net, src.data[0])

    # sequencer. don't run if program_duration is -1 though
    program_elapsed_time = time.time() - Model.program_start_time
    if Model.program_duration != -1:
        if program_elapsed_time > Model.program_duration:
            Model.next_program()

# -------
# REM CYCLE
# -------
def deepdream(net, base_img, iteration_max=10, octave_n=4, octave_scale=1.4, end='inception_4c/output', **step_params):

    # SETUPOCTAVES---
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

            # check if motion detected
            if Webcam.get().motiondetector.detection_toggle:
                Composer.ramp_toggle(True)
                Viewport.force_refresh = True
                Webcam.get().motiondetector.detection_toggle = False # reset the toggle

            # handle vieport refresh per iteration
            if Viewport.force_refresh:
                Composer.isDreaming = False # no, we'll be refreshing the frane buffer
                img = Webcam.get().read()

                # alt method - more responsive but a bit less hallucinogenic?
                Composer.send(1, caffe2rgb(Model.net, src.data[0]))

                return img

            # delegate gradient ascent to step function
            make_step(Model.net, end=end, **step_params)
            log.warning('{:02d} {:02d} {:02d}'.format(octave, i, iteration_max))

            # stored here for postproduction fx
            Composer.dreambuffer = caffe2rgb(Model.net, src.data[0])

            # write netblob to Composer - channel 0
            Composer.send(0, Composer.dreambuffer)

            # write webcam to Composer - channel 1
            Composer.send(1, Webcam.get().read())

            # send the main mix to the viewport
            Viewport.show(Composer.mix( Composer.buffer[0], Composer.buffer[1], Composer.ramp_counter))
            # Viewport.show(Composer.buffer[0])

            # attenuate step size over rem cycle
            x = step_params['step_size']
            step_params['step_size'] += x * Model.step_mult * 1.0

            # set a floor for any cyuclefx step modification
            if step_params['step_size'] < 1.1:
                step_params['step_size'] = 1.1

            i += 1

            # HUD logging
            octavemsg = '{}/{}({})'.format(octave,octave_n,Model.octave_cutoff)
            guidemsg = '({}/{}) {}'.format(Model.current_guide,len(Model.guides),Model.guides[Model.current_guide])
            iterationmsg = '{:0>3}:{:0>3} x{}'.format(i,iteration_max,Model.iteration_mult)
            stepsizemsg = '{:02.3f} x{:02.3f}'.format(step_params['step_size'],Model.step_mult)
            thresholdmsg = '{:0>6}'.format(Webcam.get().motiondetector.delta_trigger)
            floormsg = '{:0>6}'.format(Webcam.get().motiondetector.floor)
            gammamsg = '{}'.format(Webcam.get().gamma)
            intervalmsg = '{:01.2f}/{:01.2f}'.format(round(time.time() - Model.program_start_time, 2), Model.program_duration)
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
            update_HUD_log('gamma',gammamsg)
            update_HUD_log('interval', intervalmsg)
            update_HUD_log('runtime', round(time.time() - Model.installation_startup, 2))




        # CUTOFF
        # this turned out to be the last octave calculated in the series
        if octave == Model.octave_cutoff:
            Composer.isDreaming = True
            return caffe2rgb(Model.net, src.data[0])

        # EARLY EXIT
        # motion detected so we're ending this REM cycle
        # if Webcam.get().motiondetector.detection_toggle:
        #     Composer.ramp_toggle(True)
        #     Composer.isDreaming = False # no, we'll be refreshing the frane buffer
        #     Webcam.get().motiondetector.detection_toggle = False # reset the detection flag
        #     img = Webcam.get().read()
        #     Composer.send(1, img)
        #     return img

        # extract details produced on the current octave
        detail = src.data[0] - octave_current  # these feed into next octave presumably?

        # reduce iteration count for the next octave
        iteration_max = int(iteration_max - (iteration_max * Model.iteration_mult))

    Composer.isDreaming = True # yes, we'll be dreaming about this output again

    # return the resulting image (converted back to x,y,RGB structured matrix)
    return caffe2rgb(Model.net, src.data[0])


# -------
# MAIN
# -------
def main():

    now = time.time() # start timer

    # set GPU mode
    caffe.set_device(0)
    caffe.set_mode_gpu()

    # parameters
    iterations = Model.iterations
    stepsize = Model.stepsize_base
    octaves = Model.octaves
    octave_scale = Model.octave_scale
    jitter = 300

    # logging
    update_HUD_log('model',Model.caffemodel)
    update_HUD_log('username',Viewport.username)
    update_HUD_log('settings',Model.package_name)

    #
    Composer.ramp_start()

    # the madness begins
    img = Webcam.get().read()


    Composer.send(1, img)
    Composer.dreambuffer = img # initial camera image for starting


    while True:
        log.warning('new cycle')
        FX.set_cycle_start_time(time.time()) # register cycle start for duration_cutoff stepfx
        Viewport.show(Composer.mix(Composer.buffer[0], Composer.buffer[1], Composer.ramp_counter))

        ### handle viewport refresh per cycle
        if Composer.isDreaming == False:
            # Viewport.save_next_frame = True

            #  apply cyclefx, assuming they've been defined
            if Model.cyclefx is not None:
                for fx in Model.cyclefx:
                    if fx['name'] == 'xform_array':
                        FX.xform_array(Composer.dreambuffer, **fx['params'])

                    if fx['name'] == 'octave_scaler':
                        FX.octave_scaler(model=Model, **fx['params'])

            # kicks off rem sleep
            Composer.dreambuffer = deepdream(
                net,
                Composer.dreambuffer,
                objective=objective_L2,
                iteration_max = Model.iterations,
                octave_n = Model.octaves,
                octave_scale = Model.octave_scale,
                step_size = Model.stepsize_base,
                end = Model.end,
                feature = Model.features[Model.current_feature]
                )

            # commenting out this block allows unfiltered signal only
            if Viewport.force_refresh:
                Viewport.force_refresh = False

        # a bit later
        later = time.time()
        difference = later - now
        duration_msg = '{:.2f}s'.format(difference)
        now = time.time() # the new now

        # logging
        update_HUD_log('cycle_time',duration_msg) # HUD
        log.warning('cycle time: {}\n{}'.format(duration_msg,'-'*80))


# --------
# INIT
# --------

# --- LOGGING ---
logging.config.dictConfig(LogSettings.LOGGING_CONFIG)
log = logging.getLogger('logtest-debug')
log.setLevel(logging.WARNING)


# log.debug('debug message!')
# log.info('info message!')
# log.warning('warning message')
# log.error('error message')
# log.critical('critical message')
# sys.exit(0)

# --- HUD ---
# dictionary contains the key/values we'll be logging 
hud_log = {
    'octave': [None, None],
    'width': [None, None],
    'height': [None, None],
    'guide': [None, None],
    'layer': [None, None],
    'last': [None, None],
    'now': [None, None],
    'iteration': [None, None],
    'step_size': [None, None],
    'settings': [None, None],
    'threshold': [None, None],
    'detect': [None, None],
    'cycle_time': [None, None],
    'featuremap': [None, None],
    'model': [None, None],
    'username': [None, None],
    'scale': [None, None],
    'program': [None, None],
    'interval': [None, None],
    'runtime': [None, None],
    'floor': [None, None],
    'gamma':[None, None]
}

# opencv font and color
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GREEN = (0,255,0)
RED = (255,0,0)

# --- NEURAL NET ---
net = None # global reference to the neural network object

# --- CAMERA ---
Camera = [] # global reference to camera collection

# store the camera device ID's here. camera ID changes if webcam is unplugged
# default values are  [0,1] for 2 camera setup
Device = [0,1] # debug
w = data.capture_w  # capture width
h = data.capture_h # capture height

Camera.append(WebcamVideoStream(Device[0], w, h, portrait_alignment=False, log=update_HUD_log, flip_h=False, flip_v=False, gamma=0.7, floor=4000, threshold_filter=16).start())
# temp disable cam 2 for show setup
# Camera.append(WebcamVideoStream(Device[1], w, h, portrait_alignment=True, flip_h=False, flip_v=True, gamma=0.8).start())
Webcam = Cameras(source=Camera, current=Device[0])

# --- DISPLAY ---
Display = Display(width=w, height=h, camera=Webcam.get())
Viewport = Viewport('deepdreamvisionquest','silent', listener) # no screenshots if username 'silent'
Composer = Composer()

# --- PERFORMANCE SETTINGS AND rFX ---c
Model = Model(program_duration=-1) # seconds each program will run, -1 is manual
Model.set_program(0) # start with program[0]
FX = FX()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--username',help='twitter userid for sharing')
    args = parser.parse_args()
    if args.username:
        Viewport.username = '@{}'.format(args.username)
    main()