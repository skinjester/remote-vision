import numpy as np
import cv2

capture_w = 1280
capture_h = 720

now = 0 # timing reference updated each rem cycle
counter = 0 # has to do with the hud laout. sort of a hack


guides = []
guides.append('./img/gaudi1.jpg')
guides.append('./img/gaudi2.jpg')
guides.append('./img/house1.jpg')
guides.append('./img/eagle1.jpg')
guides.append('./img/tiger.jpg')
guides.append('./img/cat.jpg')
guides.append('./img/sax2.jpg')
guides.append('./img/bono.jpg')
guides.append('./img/rabbit2.jpg')
guides.append('./img/eyeballs.jpg')

# this is written to by rem.py at runtime so that it points to Composer.buffer1
# I'm using it like a scratchpad, but initializes to None
data_img = None 


models = {}
models['path'] = '../models'
models['cars'] = ('cars','deploy.prototxt','googlenet_finetune_web_car_iter_10000.caffemodel')
models['googlenet'] = ('bvlc_googlenet','deploy.prototxt','bvlc_googlenet.caffemodel')
models['places'] = ('googlenet_places205','deploy.prototxt','places205_train_iter_2400000.caffemodel')


layers = [
	'inception_4d/5x5_reduce',
	'conv2/3x3',
	'conv2/3x3_reduce',
	'conv2/norm2',
	'inception_3a/1x1',
	'inception_3a/3x3',
	'inception_3b/5x5',
	'inception_3b/output',
	'inception_3b/pool',
	'inception_4a/1x1',
	'inception_4a/3x3',
	'inception_4b/3x3_reduce',
	'inception_4b/5x5',
	'inception_4b/5x5_reduce',
	'inception_4b/output',
	'inception_4b/pool',
	'inception_4b/pool_proj',
	'inception_4c/1x1',
	'inception_4c/3x3',
	'inception_4c/3x3_reduce',
	'inception_4c/5x5',
	'inception_4c/5x5_reduce',
	'inception_4c/output',
	'inception_4c/pool',
	'inception_4d/3x3',
	'inception_4d/5x5',
	'inception_4d/5x5_reduce',
	'inception_4d/output',
	'inception_4d/pool',
	'inception_4e/1x1',
	'inception_4e/3x3',
	'inception_4e/3x3_reduce',
	'inception_4e/5x5',
	'inception_4e/5x5_reduce',
	'inception_4e/output',
	'inception_4e/pool',
	'inception_4e/pool_proj',
	'inception_5a/1x1',
	'inception_5a/3x3',
	'inception_5a/3x3_reduce',
	'inception_5a/5x5',
	'inception_5a/5x5_reduce',
	'inception_5a/output',
	'inception_5a/pool',
	'inception_5b/1x1',
	'inception_5b/3x3',
	'inception_5b/3x3_reduce',
	'inception_5b/5x5',
	'inception_5b/5x5_reduce',
	'inception_5b/output',
	'inception_5b/pool',
	'inception_5b/pool_proj'
]


def function1(param1='<empty>', param2=0):
	print 'param1:{} param2:{}'.format(param1,param2)


def function2(blur=3, radius=3):
	print 'blur:{} radius:{}'.format(blur,radius)

# img = Composer.buffer1
def xform_array(amount):
    def shiftfunc(n):
        return int(3 * np.sin(n/10,))
    for n in range(data_img.shape[1]): # number of rows in the image
        data_img[:, n] = np.roll(data_img[:, n], 3*shiftfunc(n))
    return data_img

# a list of programs
program = []

# defaults provided as a convenience
xform_array_default = {
	'name': 'xform_array',
	'params': {'amplitude':10, 'wavelength':100}
}

octave_scaler_default = {
	'name': 'octave_scaler',
	'params': {'step':0.05, 'min_scale':1.2, 'max_scale':1.6}
}

inception_xform_default = {
	'name': 'inception_xform',
	'params': {'scale':0.03}
}

cyclefx_default = [
	xform_array_default,
	octave_scaler_default,
	inception_xform_default,
]

median_blur_default = {
	'name': 'median_blur',
	'params': {'kernel_shape':3}
}

bilateral_filter_default = {
	'name': 'bilateral_filter',
	'params': {'radius': 5, 'sigma_color':30, 'sigma_xy': 30}
}

nd_gaussian_filter_default = {
	'name': 'nd_gaussian',
	'params': {'sigma': 0.6, 'order':0}
}

step_opacity_default = {
	'name': 'step_opacity',
	'params': {'opacity':1.0}
}

stepfx_default = [
	# median_blur_default,
	# bilateral_filter_default,
	nd_gaussian_filter_default,
	step_opacity_default
]

program.append({
	'name':'geo',
	'iterations':10,
	'step_size':3.0,
	'octaves':4,
	'octave_cutoff':4,
	'octave_scale':1.4,
	'iteration_mult':0.5,
	'step_mult':0.0,
	'model':'places',
	'layers':[
		'inception_3b/5x5',
	],
	'features':[-1,0,1],
	'cyclefx':cyclefx_default,
	'stepfx':stepfx_default
})

program.append({
	'name':'lofi-featuremap-superstep',
	'iterations':20,
	'step_size':3,
	'octaves':5,
	'octave_cutoff':4,
	'octave_scale':1.7,
	'iteration_mult':0.25,
	'step_mult':0.01,
	'model':'places',
	'layers':[
		'inception_4d/5x5_reduce',
		'inception_3b/pool',
		'inception_4a/1x1',
		'inception_4a/3x3',
		'inception_4b/3x3_reduce',
		'inception_4b/5x5',
		'inception_4b/5x5_reduce',
		'inception_4b/output',
		'inception_4b/pool',
		'inception_4b/pool_proj',
		'inception_4c/1x1',
		'inception_4c/3x3',
		'inception_4c/3x3_reduce',
		'inception_4c/5x5',
		'inception_4c/5x5_reduce',
		'inception_4c/output',
		'inception_4c/pool',
		'inception_4d/3x3',
		'inception_4d/5x5',
		'inception_4d/5x5_reduce',
		'inception_4d/output',
		'inception_4d/pool',
		'inception_4e/1x1',
		'inception_4e/3x3',
		'inception_4e/3x3_reduce',
		'inception_4e/5x5',
		'inception_4e/5x5_reduce',
		'inception_4e/output',
		'inception_4e/pool',
		'inception_4e/pool_proj',
		'inception_5a/1x1',
		'inception_5a/3x3',
		'inception_5a/3x3_reduce',
		'inception_5a/5x5',
		'inception_5a/5x5_reduce',
		'inception_5a/output',
		'inception_5a/pool',
		'inception_5b/1x1',
		'inception_5b/3x3',
		'inception_5b/3x3_reduce',
		'inception_5b/5x5',
		'inception_5b/5x5_reduce',
		'inception_5b/output',
		'inception_5b/pool',
		'inception_5b/pool_proj'
	],
	'features':range(-1,96),
	'cyclefx':[
		{
			'name': 'xform_array',
			'params': {'amplitude':50, 'wavelength':50}
		},
		{
			'name': 'octave_scaler',
			'params': {'step':0.01, 'min_scale':1.6, 'max_scale':1.8}
		},
		inception_xform_default
	],
	'stepfx':[
		{
			'name': 'nd_gaussian',
			'params': {'sigma': 0.9, 'order':0}
		}
	]
})

program.append({
	'name':'lofi-featuremap-wider',
	'iterations':10,
	'step_size':2,
	'octaves':6,
	'octave_cutoff':4,
	'octave_scale':1.5,
	'iteration_mult':0.1,
	'step_mult':0.0,
	'model':'places',
	'layers':[
		'inception_4d/5x5_reduce',
		'inception_3b/pool',
		'inception_4a/1x1',
		'inception_4a/3x3',
		'inception_4b/3x3_reduce',
		'inception_4b/5x5',
		'inception_4b/5x5_reduce',
		'inception_4b/output',
		'inception_4b/pool',
		'inception_4b/pool_proj',
		'inception_4c/1x1',
		'inception_4c/3x3',
		'inception_4c/3x3_reduce',
		'inception_4c/5x5',
		'inception_4c/5x5_reduce',
		'inception_4c/output',
		'inception_4c/pool',
		'inception_4d/3x3',
		'inception_4d/5x5',
		'inception_4d/5x5_reduce',
		'inception_4d/output',
		'inception_4d/pool',
		'inception_4e/1x1',
		'inception_4e/3x3',
		'inception_4e/3x3_reduce',
		'inception_4e/5x5',
		'inception_4e/5x5_reduce',
		'inception_4e/output',
		'inception_4e/pool',
		'inception_4e/pool_proj',
		'inception_5a/1x1',
		'inception_5a/3x3',
		'inception_5a/3x3_reduce',
		'inception_5a/5x5',
		'inception_5a/5x5_reduce',
		'inception_5a/output',
		'inception_5a/pool',
		'inception_5b/1x1',
		'inception_5b/3x3',
		'inception_5b/3x3_reduce',
		'inception_5b/5x5',
		'inception_5b/5x5_reduce',
		'inception_5b/output',
		'inception_5b/pool',
		'inception_5b/pool_proj'
	],
	'features':range(-1,96),
	'cyclefx':cyclefx_default,
	'stepfx':stepfx_default
})


program.append({
	'name':'lofi-featuremap',
	'iterations':10,
	'step_size':2,
	'octaves':4,
	'octave_cutoff':4,
	'octave_scale':1.5,
	'iteration_mult':0.5,
	'step_mult':0.0,
	'model':'places',
	'layers':[
		'inception_4d/5x5_reduce',
		'inception_3b/pool',
		'inception_4a/1x1',
		'inception_4a/3x3',
		'inception_4b/3x3_reduce',
		'inception_4b/5x5',
		'inception_4b/5x5_reduce',
		'inception_4b/output',
		'inception_4b/pool',
		'inception_4b/pool_proj',
		'inception_4c/1x1',
		'inception_4c/3x3',
		'inception_4c/3x3_reduce',
		'inception_4c/5x5',
		'inception_4c/5x5_reduce',
		'inception_4c/output',
		'inception_4c/pool',
		'inception_4d/3x3',
		'inception_4d/5x5',
		'inception_4d/5x5_reduce',
		'inception_4d/output',
		'inception_4d/pool',
		'inception_4e/1x1',
		'inception_4e/3x3',
		'inception_4e/3x3_reduce',
		'inception_4e/5x5',
		'inception_4e/5x5_reduce',
		'inception_4e/output',
		'inception_4e/pool',
		'inception_4e/pool_proj',
		'inception_5a/1x1',
		'inception_5a/3x3',
		'inception_5a/3x3_reduce',
		'inception_5a/5x5',
		'inception_5a/5x5_reduce',
		'inception_5a/output',
		'inception_5a/pool',
		'inception_5b/1x1',
		'inception_5b/3x3',
		'inception_5b/3x3_reduce',
		'inception_5b/5x5',
		'inception_5b/5x5_reduce',
		'inception_5b/output',
		'inception_5b/pool',
		'inception_5b/pool_proj'
	],
	'features':range(-1,96),
	'cyclefx':cyclefx_default,
	'stepfx':stepfx_default
})

program.append({
	'name':'hifi-featuremap',
	'iterations':20,
	'step_size':1.5,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.4,
	'iteration_mult':0.0,
	'step_mult':0.0,
	'model':'places',
	'layers':[
		'inception_4d/5x5_reduce',
		'inception_3b/pool',
		'inception_4a/1x1',
		'inception_4a/3x3',
		'inception_4b/3x3_reduce',
		'inception_4b/5x5',
		'inception_4b/5x5_reduce',
		'inception_4b/output',
		'inception_4b/pool',
		'inception_4b/pool_proj',
		'inception_4c/1x1',
		'inception_4c/3x3',
		'inception_4c/3x3_reduce',
		'inception_4c/5x5',
		'inception_4c/5x5_reduce',
		'inception_4c/output',
		'inception_4c/pool',
		'inception_4d/3x3',
		'inception_4d/5x5',
		'inception_4d/5x5_reduce',
		'inception_4d/output',
		'inception_4d/pool',
		'inception_4e/1x1',
		'inception_4e/3x3',
		'inception_4e/3x3_reduce',
		'inception_4e/5x5',
		'inception_4e/5x5_reduce',
		'inception_4e/output',
		'inception_4e/pool',
		'inception_4e/pool_proj',
		'inception_5a/1x1',
		'inception_5a/3x3',
		'inception_5a/3x3_reduce',
		'inception_5a/5x5',
		'inception_5a/5x5_reduce',
		'inception_5a/output',
		'inception_5a/pool',
		'inception_5b/1x1',
		'inception_5b/3x3',
		'inception_5b/3x3_reduce',
		'inception_5b/5x5',
		'inception_5b/5x5_reduce',
		'inception_5b/output',
		'inception_5b/pool',
		'inception_5b/pool_proj'
	],
	'features':range(-1,96),
	'cyclefx':cyclefx_default,
	'stepfx':stepfx_default
})

program.append({
	'name':'hifi-featuremap-iter-high',
	'iterations':100,
	'step_size':1.5,
	'octaves':4,
	'octave_cutoff':4,
	'octave_scale':1.4,
	'iteration_mult':0.5,
	'step_mult':0.01,
	'model':'places',
	'layers':[
		'inception_4d/5x5_reduce',
		'inception_3b/pool',
		'inception_4a/1x1',
		'inception_4a/3x3',
		'inception_4b/3x3_reduce',
		'inception_4b/5x5',
		'inception_4b/5x5_reduce',
		'inception_4b/output',
		'inception_4b/pool',
		'inception_4b/pool_proj',
		'inception_4c/1x1',
		'inception_4c/3x3',
		'inception_4c/3x3_reduce',
		'inception_4c/5x5',
		'inception_4c/5x5_reduce',
		'inception_4c/output',
		'inception_4c/pool',
		'inception_4d/3x3',
		'inception_4d/5x5',
		'inception_4d/5x5_reduce',
		'inception_4d/output',
		'inception_4d/pool',
		'inception_4e/1x1',
		'inception_4e/3x3',
		'inception_4e/3x3_reduce',
		'inception_4e/5x5',
		'inception_4e/5x5_reduce',
		'inception_4e/output',
		'inception_4e/pool',
		'inception_4e/pool_proj',
		'inception_5a/1x1',
		'inception_5a/3x3',
		'inception_5a/3x3_reduce',
		'inception_5a/5x5',
		'inception_5a/5x5_reduce',
		'inception_5a/output',
		'inception_5a/pool',
		'inception_5b/1x1',
		'inception_5b/3x3',
		'inception_5b/3x3_reduce',
		'inception_5b/5x5',
		'inception_5b/5x5_reduce',
		'inception_5b/output',
		'inception_5b/pool',
		'inception_5b/pool_proj'
	],
	'features':range(-1,96),
	'cyclefx':cyclefx_default,
	'stepfx':stepfx_default
})



program.append({
	'name':'wild',
	'iterations':40,
	'step_size':1.0,
	'octaves':8,
	'octave_cutoff':6,
	'octave_scale':1.2,
	'iteration_mult':0.1,
	'step_mult':0.00,
	'model':'places',
	'layers':[
		'inception_4c/output',
		'inception_4d/pool',
		'inception_4e/1x1',
		'inception_4e/3x3',
		'inception_4e/3x3_reduce',
		'inception_4e/5x5',
		'inception_4e/5x5_reduce',
		'inception_4e/output',
		'inception_4e/pool',
		'inception_4e/pool_proj'
	],
	'features':range(-1,96),
	'cyclefx':cyclefx_default,
	'stepfx':stepfx_default
})

program.append({
	'name':'libraryofbabel',
	'iterations':30,
	'step_size':2.0,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.4,
	'iteration_mult':0.0,
	'step_mult':0.01,
	'model':'places',
	'layers':[
		'inception_4d/5x5_reduce',
		'conv2/3x3',
		'conv2/3x3_reduce',
		'conv2/norm2',
		'inception_3a/1x1',
		'inception_3a/3x3',
		'inception_3b/5x5',
		'inception_3b/output',
		'inception_3b/pool',
		'inception_4a/1x1',
		'inception_4a/3x3',
		'inception_4b/3x3_reduce',
		'inception_4b/5x5',
		'inception_4b/5x5_reduce',
		'inception_4b/output',
		'inception_4b/pool',
		'inception_4b/pool_proj',
		'inception_4c/1x1',
		'inception_4c/3x3',
		'inception_4c/3x3_reduce',
		'inception_4c/5x5',
		'inception_4c/5x5_reduce',
		'inception_4c/output',
		'inception_4c/pool',
		'inception_4d/3x3',
		'inception_4d/5x5',
		'inception_4d/5x5_reduce',
		'inception_4d/output',
		'inception_4d/pool',
		'inception_4e/1x1',
		'inception_4e/3x3',
		'inception_4e/3x3_reduce',
		'inception_4e/5x5',
		'inception_4e/5x5_reduce',
		'inception_4e/output',
		'inception_4e/pool',
		'inception_4e/pool_proj',
		'inception_5a/1x1',
		'inception_5a/3x3',
		'inception_5a/3x3_reduce',
		'inception_5a/5x5',
		'inception_5a/5x5_reduce',
		'inception_5a/output',
		'inception_5a/pool',
		'inception_5b/1x1',
		'inception_5b/3x3',
		'inception_5b/3x3_reduce',
		'inception_5b/5x5',
		'inception_5b/5x5_reduce',
		'inception_5b/output',
		'inception_5b/pool',
		'inception_5b/pool_proj'
	],
	'features':range(-1,96),
	'cyclefx':cyclefx_default,
	'stepfx':stepfx_default
})

program.append({
	'name':'smallerlibraryofbabel',
	'iterations':50,
	'step_size':4.0,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.2,
	'iteration_mult':0.0,
	'step_mult':0.00,
	'model':'places',
	'layers':[
		'inception_4d/5x5_reduce',
		'conv2/3x3',
		'conv2/3x3_reduce',
		'conv2/norm2',
		'inception_3a/1x1',
		'inception_3a/3x3',
		'inception_3b/5x5',
		'inception_3b/output',
		'inception_3b/pool',
		'inception_4a/1x1',
		'inception_4a/3x3',
		'inception_4b/3x3_reduce',
		'inception_4b/5x5',
		'inception_4b/5x5_reduce',
		'inception_4b/output',
		'inception_4b/pool',
		'inception_4b/pool_proj',
		'inception_4c/1x1',
		'inception_4c/3x3',
		'inception_4c/3x3_reduce',
		'inception_4c/5x5',
		'inception_4c/5x5_reduce',
		'inception_4c/output',
		'inception_4c/pool',
		'inception_4d/3x3',
		'inception_4d/5x5',
		'inception_4d/5x5_reduce',
		'inception_4d/output',
		'inception_4d/pool',
		'inception_4e/1x1',
		'inception_4e/3x3',
		'inception_4e/3x3_reduce',
		'inception_4e/5x5',
		'inception_4e/5x5_reduce',
		'inception_4e/output',
		'inception_4e/pool',
		'inception_4e/pool_proj',
		'inception_5a/1x1',
		'inception_5a/3x3',
		'inception_5a/3x3_reduce',
		'inception_5a/5x5',
		'inception_5a/5x5_reduce',
		'inception_5a/output',
		'inception_5a/pool',
		'inception_5b/1x1',
		'inception_5b/3x3',
		'inception_5b/3x3_reduce',
		'inception_5b/5x5',
		'inception_5b/5x5_reduce',
		'inception_5b/output',
		'inception_5b/pool',
		'inception_5b/pool_proj'
	],
	'features':range(-1,96),
	'cyclefx':cyclefx_default,
	'stepfx':stepfx_default
})

program.append({
	'name':'ghost1',
	'iterations':10,
	'step_size':4.0,
	'octaves':6,
	'octave_cutoff':4,
	'octave_scale':1.5,
	'iteration_mult':0.0,
	'step_mult':0.05,
	'model':'places',
	'layers':[
		'inception_3b/pool',
		'inception_5a/3x3_reduce',
		'inception_5a/5x5'
	],
	'features':range(-1,96),
	'cyclefx':cyclefx_default,
	'stepfx':stepfx_default
})

program.append({
	'name':'ghost2',
	'iterations':10,
	'step_size':2.0,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.4,
	'iteration_mult':0.0,
	'step_mult':0.01,
	'model':'places',
	'layers':[
		'inception_3b/output',
	],
	'features':[27],
	'cyclefx':cyclefx_default,
	'stepfx':stepfx_default
})


program.append({
	'name':'ghost3',
	'iterations':20,
	'step_size':2.0,
	'octaves':8,
	'octave_cutoff':5,
	'octave_scale':1.4,
	'iteration_mult':0.0,
	'step_mult':0.01,
	'model':'places',
	'layers':[
		'inception_3a/3x3',
	],
	'features':[21],
	'cyclefx':cyclefx_default,
	'stepfx':stepfx_default
})

program.append({
	'name':'ghost4',
	'iterations':10,
	'step_size':2.0,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.5,
	'iteration_mult':0.0,
	'step_mult':0.01,
	'model':'places',
	'layers':[
		'inception_3b/5x5',
	],
	'features':[21],
	'cyclefx':cyclefx_default,
	'stepfx':stepfx_default
})

program.append({
	'name':'ghost5',
	'iterations':10,
	'step_size':2.0,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.5,
	'iteration_mult':0.0,
	'step_mult':0.01,
	'model':'places',
	'layers':[
		'inception_4e/output',
	],
	'features':[24],
	'cyclefx':cyclefx_default,
	'stepfx':stepfx_default
})

program.append({
	'name':'ghost6',
	'iterations':10,
	'step_size':2.0,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.5,
	'iteration_mult':0.0,
	'step_mult':0.01,
	'model':'places',
	'layers':[
		'inception_4a/3x3',
	],
	'features':[15],
	'cyclefx':cyclefx_default,
	'stepfx':stepfx_default
})

program.append({
	'name':'ghost7',
	'iterations':10,
	'step_size':2.0,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.5,
	'iteration_mult':0.0,
	'step_mult':0.01,
	'model':'places',
	'layers':[
		'inception_4a/3x3',
	],
	'features':[12],
	'cyclefx':cyclefx_default,
	'stepfx':stepfx_default
})

program.append({
	'name':'ghost8',
	'iterations':10,
	'step_size':2.0,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.5,
	'iteration_mult':0.0,
	'step_mult':0.01,
	'model':'places',
	'layers':[
		'inception_3b/output',
	],
	'features':[11],
	'cyclefx':cyclefx_default,
	'stepfx':stepfx_default
})

program.append({
	'name':'ghost9',
	'iterations':10,
	'step_size':2.0,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.5,
	'iteration_mult':0.0,
	'step_mult':0.01,
	'model':'places',
	'layers':[
		'inception_4d/5x5_reduce',
	],
	'features':[11],
	'cyclefx':cyclefx_default,
	'stepfx':stepfx_default
})

program.append({
	'name':'ghost10',
	'iterations':10,
	'step_size':2.0,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.5,
	'iteration_mult':0.0,
	'step_mult':0.01,
	'model':'places',
	'layers':[
		'inception_4d/5x5_reduce',
	],
	'features':[15],
	'cyclefx':cyclefx_default,
	'stepfx':stepfx_default
})

program.append({
	'name':'ghost11',
	'iterations':10,
	'step_size':2.0,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.5,
	'iteration_mult':0.0,
	'step_mult':0.01,
	'model':'places',
	'layers':[
		'inception_4d/5x5_reduce',
	],
	'features':[17],
	'cyclefx':cyclefx_default,
	'stepfx':stepfx_default
})

program.append({
	'name':'ghost12',
	'iterations':10,
	'step_size':2.0,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.5,
	'iteration_mult':0.0,
	'step_mult':0.01,
	'model':'places',
	'layers':[
		'inception_4d/5x5_reduce',
	],
	'features':[2],
	'cyclefx':cyclefx_default,
	'stepfx':stepfx_default
})

program.append({
	'name':'ghost13',
	'iterations':10,
	'step_size':2.0,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.5,
	'iteration_mult':0.0,
	'step_mult':0.01,
	'model':'places',
	'layers':[
		'inception_4d/5x5_reduce',
	],
	'features':[11],
	'cyclefx':cyclefx_default,
	'stepfx':stepfx_default
})



program.append({
	'name':'lofi',
	'iterations':10,
	'step_size':8.0,
	'octaves':4,
	'octave_cutoff':4,
	'octave_scale':1.4,
	'iteration_mult':0.5,
	'step_mult':0.0,
	'model':'places',
	'layers':[
		'inception_4c/output'
	],
	'features':[1],
	'cyclefx':cyclefx_default,
	'stepfx':stepfx_default
})
