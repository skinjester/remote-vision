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


models = {
	'path': '../models',
	'cars': ('cars','deploy.prototxt','googlenet_finetune_web_car_iter_10000.caffemodel'),
	'googlenet': ('bvlc_googlenet','deploy.prototxt','bvlc_googlenet.caffemodel'),
	'places': ('googlenet_places205','deploy.prototxt','places205_train_iter_2400000.caffemodel')
}


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

# a list of programs
program = []

# defaults provided as a convenience
xform_array_default = {
	'name': 'xform_array',
	'params': {'amplitude':20, 'wavelength':100}
}

octave_scaler_default = {
	'name': 'octave_scaler',
	'params': {'step':0.01, 'min_scale':1.6, 'max_scale':1.7}
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

duration_cutoff_default = {
	'name': 'duration_cutoff',
	'params': {'duration':2.0}
}

stepfx_default = [
	# median_blur_default,
	bilateral_filter_default,
	# nd_gaussian_filter_default,
	# step_opacity_default,
	# duration_cutoff_default
]

program.append({
	'name':'Shpongled',
	'iterations':20,
	'step_size':2,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.7,
	'iteration_mult':0.0,
	'step_mult':0.01,
	'model':'googlenet',
	'layers':[
		'inception_4b/5x5_reduce'
	],
	'features':[7],
	'cyclefx':[
		{
			'name': 'octave_scaler',
			'params': {'step':0.01, 'min_scale':1.5, 'max_scale':1.7}
		},
		inception_xform_default
	],
	'stepfx':[
		{
			'name': 'nd_gaussian',
			'params': {'sigma': 0.4, 'order':0}
		},
		{
		'name': 'bilateral_filter',
		'params': {'radius': 7, 'sigma_color':30, 'sigma_xy': 60}
		},
		{
			'name': 'duration_cutoff',
			'params': {'duration':10.0}
		}
	]
})


program.append({
	'name':'world of aspiration',
	'iterations':50,
	'step_size':3,
	'octaves':4,
	'octave_cutoff':4,
	'octave_scale':1.5,
	'iteration_mult':0.1,
	'step_mult':0.01,
	'model':'googlenet',
	'layers':[
		'inception_4e/5x5_reduce',
	],
	'features':[30],
	'cyclefx':[
		{
			'name': 'octave_scaler',
			'params': {'step':0.02, 'min_scale':1.2, 'max_scale':1.6}
		},
		{
			'name': 'inception_xform',
			'params': {'scale':0.05}
		},
		xform_array_default
	],
	'stepfx':[
		# {
		# 	'name': 'nd_gaussian',
		# 	'params': {'sigma': 0.5, 'order':0}
		# },
		{
			'name': 'median_blur',
			'params': {'kernel_shape':5}
		},
		# {
		# 	'name': 'bilateral_filter',
		# 	'params': {'radius': 10, 'sigma_color':30, 'sigma_xy': 30}
		# }
	]
})

# ___ Initial program ___

program.append({
	'name':'geo',
	'iterations':10,
	'step_size':3.0,
	'octaves':4,
	'octave_cutoff':4,
	'octave_scale':1.4,
	'iteration_mult':0.0,
	'step_mult':0.0,
	'model':'googlenet',
	'layers':[
		'inception_3b/5x5',
	],
	'features':[-1],
	'cyclefx':cyclefx_default,
	'stepfx':[
		bilateral_filter_default,
		{
			'name': 'duration_cutoff',
			'params': {'duration':2.0}
		}
	]
})

