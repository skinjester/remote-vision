# TODO: not needing all of these imports. cleanup
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

class Amplifier(object):
    def __init__(self):
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

    def set_program(self, key):
        self.iterations = data.program[key]['iterations']
        self.stepsize_base = data.program[key]['step_size']
        self.octaves = data.program[key]['octaves']
        self.octave_cutoff = data.program[key]['octave_cutoff']
        self.octave_scale = data.program[key]['octave_scale']
        self.iteration_mult = data.program[key]['iteration_mult']
        self.step_mult = data.program[key]['step_mult']
        self.package_name = key

class Model(object):
    def __init__(self, modelkey='googlenet', current_layer=0):
        self.guide_features = None
        self.net = None
        self.net_fn = None
        self.param_fn = None
        self.caffemodel = None
        self.end = None
        self.models = data.models
        self.guides = data.guides
        self.features = data.features
        self.current_guide = 0
        self.current_layer = current_layer
        self.current_feature = 0
        self.layers = data.layers
        self.first_time_through = True

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

        self.choose_model(modelkey)
        self.set_endlayer(self.layers[self.current_layer])
        self.set_featuremap(self.features[self.current_feature])


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
            self.param_fn, mean=np.float32([104.0, 116.0, 122.0]),
            channel_swap=(2, 1, 0))

        

    def set_program(self, name):
        self.iterations = data.program[name]['iterations']
        self.stepsize_base = data.program[name]['step_size']
        self.octaves = data.program[name]['octaves']
        self.octave_cutoff = data.program[name]['octave_cutoff']
        self.octave_scale = data.program[name]['octave_scale']
        self.iteration_mult = data.program[name]['iteration_mult']
        self.step_mult = data.program[name]['step_mult']
        self.package_name = name

    def guide_image(self):
        # current guide img
        guide = np.float32(PIL.Image.open(self.guides[self.current_guide]))

        #  pick some target layer and extract guide image features
        h, w = guide.shape[:2]
        src, dst = self.net.blobs['data'], self.net.blobs[self.end]
        src.reshape(1, 3, h, w)
        src.data[0] = rgb2caffe(self.net, guide)
        self.net.forward(end=self.end)
        self.guide_features = dst.data[0].copy()
        MotionDetector.wasMotionDetected = True  # force refresh

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
        MotionDetector.is_paused = False

    def set_endlayer(self,end):
        self.end = end
        Viewport.force_refresh = True
        print '######## [Model] new layer {}/featuremaps={}'.format(end,self.net.blobs[self.end].data.shape[1])
        update_log('layer','{}/{}'.format(end,self.net.blobs[self.end].data.shape[1]))


    def set_featuremap(self,index):
        self.feature_ID = self.features[index]
        Viewport.force_refresh = True
        print '######## [Model] new featuremap {}'.format(self.feature_ID)
        update_log('featuremap',self.feature_ID)

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

    def prev_feature(self):
        self.current_feature -= 1
        if self.current_feature < 0:
            self.current_feature = len(self.features)-1
        self.set_featuremap(self.current_feature)

    def next_feature(self):
        self.current_feature += 1
        if self.current_feature > len(self.features)-1:
            self.current_feature = 0
        self.set_featuremap(self.current_feature)

    def get_feature_ID(self):
        return self.features[self.current_feature]


 
class Viewport(object):

    def __init__(self, window_name, username, listener):
        self.window_name = window_name
        self.viewport_w = data.viewport_size[0]
        self.viewport_h = data.viewport_size[1]
        self.b_show_HUD = False
        self.keypress_mult = 0 # accelerate value changes when key held
        self.b_show_stats = False
        self.motiondetect_log_enabled = False
        self.blend_ratio = 0.0
        self.save_next_frame = False
        self.username = username
        self.listener = listener
        self.force_refresh = False
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    
    def show(self, image):
        # convert and clip floating point matrix into RGB bounds as integers
        image = np.uint8(np.clip(image, 0, 255))

        # resize image to fit viewport, skip if already at full size
        if image.shape[1] != data.viewport_size[0]:
            image = cv2.resize(image,
                (data.viewport_size[0], data.viewport_size[1]),
                interpolation = cv2.INTER_LINEAR)

        image = Composer.update(image)

        image = self.postfx(image) # HUD
        if self.b_show_stats:
            image = self.postfx2(image) # stats
        cv2.imshow(self.window_name, image)

        self.monitor()
        self.listener(image) # refresh display

        # export image if condition is met
       # if someFlag:
            #someFlag = False
            #self.export(image)

    def export(self, image):
        pass
        # self.save_next_frame = True
        # #print '[main] save rendered frame'
        # Viewport.save_next_frame = False
        # make_sure_path_exists(Viewport.username)
        # export_path = '{}/{}.jpg'.format(Viewport.username,time.time())
        # savefile = cv2.cvtColor(Composer.buffer1, cv2.COLOR_BGR2RGB)
        # PIL.Image.fromarray(np.uint8(savefile)).save(export_path)
        #tweet(export_path)

    def postfx(self, image):
        if self.b_show_HUD:
            image = show_HUD(image)
        return image

    def postfx2(self, image):
        image = show_stats(image)
        return image
    
    def monitor(self):
        if self.motiondetect_log_enabled:
            cv2.imshow('delta', MotionDetector.t_delta_framebuffer)

    def shutdown(self):
        sys.exit()

class Composer(object):

    def __init__(self):
        self.is_dirty = False # the type of frame in buffer1. dirty when recycling clean when refreshing
        self.is_new_cycle = True
        self.buffer1 = np.zeros((data.capture_size[1], data.capture_size[0] ,3), np.uint8) # uses camera capture dimensions
        self.buffer2 = np.zeros((data.viewport_size[1], data.viewport_size[0], 3), np.uint8) # uses camera capture dimensions
        self.opacity = 1.0
        self.is_compositing_enabled = False
        self.xform_scale = 0.09

    def update(self, image):
        if self.is_dirty: 
            print '[Composer] recycle'
            '''
            if self.is_new_cycle:
                #print '[Composer] inception'
                self.buffer1 = inceptionxform(image, self.xform_scale, data.capture_size)
                #print '[Composer] xform scale {}'.format(self.xform_scale)

            self.is_dirty = False
            self.is_compositing_enabled = False
            '''

        else:
            #print '[Composer] refresh'
            if self.is_new_cycle and MotionDetector.isResting() == False:
                #print '[Composer] compositing enabled'
                self.is_compositing_enabled = True

            if self.is_compositing_enabled:
                #print '[Composer] compositing buffer1:{} buffer2:{}'.format(image.shape,self.buffer2.shape)
                image = cv2.addWeighted(self.buffer2, self.opacity, image, 1-self.opacity, 0, image)
                self.opacity = self.opacity * 0.9
                if self.opacity <= 0.1:
                    self.opacity = 1.0
                    self.is_compositing_enabled = False
                    #print '[Composer] stopped compositing'

        return image

    def write_buffer2(self,image):
        if self.is_compositing_enabled == False:
            # convert and clip floating point matrix into RGB bounds
            self.buffer2 = np.uint8(np.clip(image, 0, 255))

            ### resize buffer 2 to match viewport dimensions
            if image.shape[1] != data.viewport_size[0]:
                print '[Composer][write_buffer2] resize buffer2 to viewport'
                self.buffer2 = cv2.resize(self.buffer2, (data.viewport_size[0], data.viewport_size[1]), interpolation = cv2.INTER_LINEAR)
        return

    # def write(self, buffer=1, rgbimage):
    #     if

def inceptionxform(image,scale,capture_size):
    return nd.affine_transform(image, [1-scale, 1, 1], [capture_size[1]*scale/2, 0, 0], order=1)
    #return nd.affine_transform(image, [1-scale, 1-scale, 1], [capture_size[1]*scale/2, capture_size[0]*scale/2, 0], order=1)

def iterationPostProcess(net_data_blob):
    return blur(net_data_blob, 0.5)

def blur(img, sigma):
    if sigma > 0:
        img = nd.filters.gaussian_filter(img, sigma, order=0)
    return img

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

def update_log(key,new_value):
    log[key] = '{}'.format(new_value)

def show_stats(image):
    stats_overlay = image.copy()
    opacity = 0.9
    cv2.putText(stats_overlay, 'show_stats()', (30, 40), font, 1.0, white)
    return cv2.addWeighted(stats_overlay, opacity, image, 1-opacity, 0, image)

def show_HUD(image):
    # rectangle
    overlay = image.copy()
    opacity = 0.5
    cv2.rectangle(overlay,(0,0),(data.viewport_size[0], data.viewport_size[1]), (0, 0, 0), -1)
    #cv2.rectangle(image_to_draw_on, (x1,y1), (x2,y2), (r,g,b), line_width )

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
    write_Text('featuremap')
    write_Text('guide')
    write_Text('width')
    write_Text('height')
    write_Text('octave')
    write_Text('iteration')
    write_Text('step_size')
    write_Text('rem_cycle')

    # add overlay back to source
    return cv2.addWeighted(overlay, opacity, image, 1-opacity, 0, image)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

# keyboard event handler
def listener(image): # yeah... passing image as a convenience
    key = cv2.waitKey(1) & 0xFF
    #print '[listener] key:{}'.format(key)

    # Escape key: Exit
    if key == 27:
        print '[listener] shutdown'
        Viewport.shutdown()

    # ENTER key: save picture
    elif key==13:
        print '[listener] save'
        Viewport.export(image)

    # `(tilde) key: toggle HUD
    elif key == 96:
        Viewport.b_show_HUD = not Viewport.b_show_HUD
        print '[listener] HUD'

    # + key (numpad): increase motion threshold
    elif key == 171:
        Viewport.keypress_mult +=1
        MotionDetector.delta_trigger += (1000 + (200 * Viewport.keypress_mult))
        Viewport.b_show_stats = True
        print '[listener] delta_trigger ++ {}'.format(MotionDetector.delta_trigger)

    # - key (numpad) : decrease motion threshold    
    elif key == 173: 
        Viewport.keypress_mult +=1
        MotionDetector.delta_trigger -= (1000 + (100 * Viewport.keypress_mult))
        if MotionDetector.delta_trigger < 1:
            MotionDetector.delta_trigger = 1
        Viewport.b_show_stats = True
        print '[listener] delta_trigger -- {}'.format(MotionDetector.delta_trigger)

    # , key : previous featuremap    
    elif key == 44:
        print '[listener] previous featuremap'
        Model.prev_feature()

    # . key : next featuremap    
    elif key == 46:
        print '[listener] next featuremap'
        Model.next_feature()

    # 1 key : toggle motion detect window
    elif key == 49: 
        Viewport.motiondetect_log_enabled = not Viewport.motiondetect_log_enabled
        if Viewport.motiondetect_log_enabled:
            cv2.namedWindow('delta',cv2.WINDOW_AUTOSIZE)
        else:
            cv2.destroyWindow('delta')   
        print '[keylistener] motion detect monitor: {}'.format(Viewport.motiondetect_log_enabled)

    # p key : pause/unpause motion detection    
    elif key == 112:
        MotionDetector.is_paused = not MotionDetector.is_paused
        print '[listener] pause motion detection {}'.format(MotionDetector.is_paused)
        if MotionDetector.is_paused:
            MotionDetector.delta_trigger_history = MotionDetector.delta_trigger
            MotionDetector.delta_trigger = data.viewport_size[0] * data.viewport_size[1]
            MotionDetector.wasMotionDetected = False
            MotionDetector.wasMotionDetected_history = False
            MotionDetector.timer_enabled = False
            Viewport.delta_count = 0
            Viewport.delta_count_history = 0
        else:
            MotionDetector.delta_trigger = MotionDetector.delta_trigger_history

    # x key: previous network layer
    elif key == 120:
        print '>>> [listener] next layer'
        Model.next_layer()

    # z key: next network layer
    elif key == 122:
        print '<<< [listener] previous layer'
        Model.prev_layer()

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
    dst.diff[0].reshape(ch,-1)[:] = y[:,A.argmax(1)] # select one sthta match best



# -------
# implements forward and backward passes thru the network
# apply normalized ascent step upon the image in the networks data blob
# ------- 
'''
def make_step(net, step_size=1.5, end='inception_4c/output',jitter=32, clip=True, objective=objective_L2, feature=1):
    print '[make_step] wasMotionDetected {}'.format(MotionDetector.wasMotionDetected)
    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end] # destination is the end layer specified by argument

    ox, oy = np.random.randint(-jitter, jitter + 1, 2)          # calculate jitter
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift

    # this bit is where the neural net runs the hallucination
    net.forward(end=end)    # make sure we stop on the chosen neural layer
    objective(dst)          # specify the optimization objective
    net.backward(start=end) # backwards propagation
    g = src.diff[0]         # store the error 

    # apply normalized ascent step to the image array 
    src.data[:] += step_size / np.abs(g).mean() * g

    # unshift image jitter              
    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2)   

    # subtract image mean and clip our matrix to the values
    bias = net.transformer.mean['data']
    src.data[:] = np.clip(src.data, -bias, 255-bias)

    # postprocess (blur) this iteration
    src.data[0] = iterationPostProcess(src.data[0])
'''



# -------
# implements forward and backward passes thru the network
# apply normalized ascent step upon the image in the networks data blob
# supports Feature Map activation
# ------- 
def make_step(net, step_size=1.5, end='inception_4c/output', jitter=32, clip=True, feature=-1):

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2)
            
    net.forward(end=end)
        
    if feature == -1:
        dst.diff[:] = dst.data
    else:
        dst.diff.fill(0.0)
        dst.diff[0,feature,:] = dst.data[0,feature,:]

    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    m = np.abs(g).mean()

    if m > 0.0:
        src.data[:] += step_size/m * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
            
    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)

    # postprocess (blur) this iteration
    src.data[0] = iterationPostProcess(src.data[0])

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

    # GRB: Not seeing this condition get hit after several minutes of observation    
    if MotionDetector.wasMotionDetected:
        Composer.write_buffer2(which_camera.read()[1])
        Composer.is_dirty = False # no, we'll be refreshing the frane buffer
        print '!!!! [deepdream] abort return camera'
        return which_camera.read()[1] 
        #print '[deepdream] was.MotionDetected TRUE'
        #return which_camera.read()[1]
        #return np.zeros((data.capture_size[1], data.capture_size[0] ,3), np.uint8)

    # SETUPOCTAVES---
    Composer.is_new_cycle = False
    src = Model.net.blobs['data']
    octaves = [rgb2caffe(Model.net, base_img)]
    for i in xrange(octave_n - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0 / octave_scale, 1.0 / octave_scale), order=1))
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

        Amplifier.stepsize = Amplifier.stepsize_base # reset step size to default each octave
        step_params['step_size'] = Amplifier.stepsize # modifying the **step_params list for makestep

        # OCTAVECYCLE
        i=0
        while i < iteration_max:
            MotionDetector.process()
            if not MotionDetector.isResting():
                break

            # delegate gradient ascent to step function
            print '{:02d}:{:03d}:{:03d}'.format(octave,i,iteration_max)
            make_step(Model.net, end=end, **step_params)

            # write netblob to Composer
            Composer.buffer1 = caffe2rgb(Model.net, src.data[0])
            #Composer.buffer1 = Composer.buffer1 * (255.0 / np.percentile(Composer.buffer1, 99.98)) # normalize contrast
            Viewport.show(Composer.buffer1)

            # attenuate step size over rem cycle
            x = step_params['step_size']
            step_params['step_size'] += x * Amplifier.step_mult * 1.0

            i += 1

            # logging
            octavemsg = '{}/{}({})'.format(octave,octave_n,Amplifier.octave_cutoff)
            guidemsg = '({}/{}) {}'.format(Model.current_guide,len(Model.guides),Model.guides[Model.current_guide])
            iterationmsg = '{:0>3}/{:0>3}({})'.format(i,iteration_max,Amplifier.iteration_mult)
            stepsizemsg = '{:02.3f}({:02.3f})'.format(step_params['step_size'],Amplifier.step_mult)
            thresholdmsg = '{:0>6}'.format(MotionDetector.delta_trigger)
            update_log('octave',octavemsg)
            update_log('width',w)
            update_log('height',h)
            update_log('guide',guidemsg)
            #update_log('layer',end)
            update_log('iteration',iterationmsg)
            update_log('step_size',stepsizemsg)
            update_log('settings',Amplifier.package_name)
            update_log('threshold',thresholdmsg)

        # probably temp? export each completed iteration
        # Viewport.export(Composer.buffer1)


        # AMPLIFIER CUTOFF
        # this turned out to be the last octave calculated in the series
        if octave == Amplifier.octave_cutoff:
            Composer.is_dirty = True
            print '[deepdream] {:02d}:{:03d}:{:03d} amplifier cutoff return net blob {}'.format(octave,i,iteration_max,src.data[0].shape)
            return caffe2rgb(
                Model.net, src.data[0])

        # EARLY EXIT
        # motion detected so we're ending this REM cycle
        if MotionDetector.wasMotionDetected:
            Composer.write_buffer2(
                caffe2rgb(Model.net, src.data[0]))
            Composer.is_dirty = False # no, we'll be refreshing the frane buffer
            print '[deepdream] early exit return camera'
            return which_camera.read()[1] 

        # extract details produced on the current octave
        detail = src.data[0] - octave_current

        # reduce iteration count for the next octave
        iteration_max = iteration_max - int(iteration_max * Amplifier.iteration_mult)
        #iteration_max = Amplifier.next_iteration(iteration_max)

    # return the resulting image (converted back to x,y,RGB structured matrix)
    #print '[deepdream] {:02d}:{:03d}:{:03d} return net blob'.format(octave,i,iteration_max)
    Composer.is_dirty = True # yes, we'll be recycling the Composer

    # export finished img
    return caffe2rgb(Model.net, src.data[0])

# -------
# MAIN
# ------- 
def main():

    # start timer
    now = time.time()
    cycle = 2

    # set GPU mode
    caffe.set_device(0)
    caffe.set_mode_gpu()

    # parameters
    #Amplifier.set_program('hirez-fast')
    iterations = Amplifier.iterations
    stepsize = Amplifier.stepsize_base
    octaves = Amplifier.octaves
    octave_scale = Amplifier.octave_scale
    jitter = 300
    update_log('model',Model.caffemodel)
    update_log('username',Viewport.username)
    update_log('settings',Amplifier.package_name)


    # the madness begins 
    Composer.buffer1 = which_camera.read()[1] # initial camera image for init
    while True:
        print 'new cycle'
        Composer.is_new_cycle = True
        Viewport.show(Composer.buffer1)
        MotionDetector.process()
        if MotionDetector.wasMotionDetected:
            Composer.is_dirty = False
        # octave_scale += 0.2 * cycle
        # if octave_scale > 1.8 or octave_scale < 1.2:
        #     cycle = -1 * cycle
        # #print '[main] octave_scale {0:5.2f}'.format(octave_scale)

        print '[main] Composer.is_dirty {}'.format(Composer.is_dirty)
        if Composer.is_dirty == False or Viewport.force_refresh:
            # kicks off rem sleep - will begin continual iteration of the image through the model
            Composer.buffer1 = deepdream(net, Composer.buffer1, iteration_max = iterations, octave_n = octaves, octave_scale = octave_scale, step_size = stepsize, end = Model.end, feature = Model.feature_ID)
            Viewport.force_refresh = False

        
        #print '[main] !!! Composer.buffer1 shape is {}'.format(Composer.buffer1.shape)

        # a bit later
        later = time.time()
        difference = later - now
        print '[main] end cycle: {0:5.2f}s'.format(difference)
        print '-'*20
        duration_msg = '{}s'.format(difference)
        update_log('rem_cycle',duration_msg)
        now = time.time()

        # export each finished img to filesystem
        #Viewport.export(Composer.buffer1)

# -------- 
# INIT
# --------
net = None

# HUD
# dictionary contains the key/values we'll be logging
font = cv2.FONT_HERSHEY_SIMPLEX
white = (255, 255, 255)

log = {
    'octave': None,
    'width': None,
    'height': None,
    'guide': None,
    'layer': None,
    'last': None,
    'now': None,
    'iteration': None,
    'step_size': None,
    'settings': None,
    'threshold': None,
    'detect': None,
    'rem_cycle': None
}

# set global camera object to input dimensions
which_camera = cv2.VideoCapture(0)
which_camera.set(3, data.capture_size[0])
which_camera.set(4, data.capture_size[1])

MotionDetector = MotionDetector(16000, which_camera, update_log)
Viewport = Viewport('deepdreamvisionquest','@deepdreamvisionquest', listener)
Composer = Composer()
Model = Model()
Amplifier = Amplifier()

# model is googlenet unless specified otherwise
#Model.choose_model('places')
#Model.choose_model('cars')
#Model.set_endlayer(data.layers[0])

Amplifier.set_program('hifi-featuremap')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--username',help='twitter userid for sharing')
    args = parser.parse_args()
    if args.username:
        Viewport.username = '@{}'.format(args.username)
    main()

