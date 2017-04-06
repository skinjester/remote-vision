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


program.append({
	'name':'spiderman',
	'iterations':10,
	'step_size':2.0,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.5,
	'iteration_mult':0.1,
	'step_mult':0.02,
	'model':'places',
	'layers':[
		'inception_4a/3x3',
	],
	'features':[12],
	'cyclefx':cyclefx_default,
	'stepfx':[
		{
			'name': 'bilateral_filter',
			'params': {'radius': 5, 'sigma_color':30, 'sigma_xy': 30}
		},
		{
			'name': 'nd_gaussian',
			'params': {'sigma': 0.4, 'order':0}
		},
	]
})

program.append({
	'name':'Bridges',
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
	'name':'Cars',
	'iterations':10,
	'step_size':2.0,
	'octaves':5,
	'octave_cutoff':4,
	'octave_scale':1.3,
	'iteration_mult':0.0,
	'step_mult':0.01,
	'model':'googlenet',
	'layers':['inception_4c/3x3_reduce'],
	'features':[11],
	'cyclefx':[
		{
			'name': 'octave_scaler',
			'params': {'step':0.05, 'min_scale':1.3, 'max_scale':1.7}
		},
		inception_xform_default,
		{
			'name': 'xform_array',
			'params': {'amplitude':10, 'wavelength':100}
		}

	],
	'stepfx':[
		{
			'name': 'nd_gaussian',
			'params': {'sigma': 0.5, 'order':0}
		},
		{
		'name': 'bilateral_filter',
		'params': {'radius': 7, 'sigma_color':10, 'sigma_xy': 10}
		},
	]
})


program.append({
	'name':'sphere',
	'iterations':60,
	'step_size':1.0,
	'octaves':5,
	'octave_cutoff':4,
	'octave_scale':1.4,
	'iteration_mult':0.0,
	'step_mult':0.01,
	'model':'googlenet',
	'layers':[
		'inception_4c/3x3_reduce',
	],
	'features':[8],
	'cyclefx':[
		{
			'name': 'octave_scaler',
			'params': {'step':0.1, 'min_scale':1.5, 'max_scale':1.7}
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
		'params': {'radius': 7, 'sigma_color':10, 'sigma_xy': 10}
		},
		{
			'name': 'duration_cutoff',
			'params': {'duration':10.0}
		},
	]
})


program.append({
	'name':'Flowers on Mars`',
	'iterations':10,
	'step_size':2.0,
	'octaves':6,
	'octave_cutoff':4,
	'octave_scale':1.5,
	'iteration_mult':0.0,
	'step_mult':0.05,
	'model':'googlenet',
	'layers':[
		'inception_3b/pool',
	],
	'features':[16],
	'cyclefx':[
		{
			'name': 'octave_scaler',
			'params': {'step':0.1, 'min_scale':1.5, 'max_scale':1.7}
		},
		inception_xform_default
	],
	'stepfx':[
		# {
		# 	'name': 'nd_gaussian',`
		# 	'params': {'sigma': 0.4, 'order':0}
		# },
		{
		'name': 'bilateral_filter',
		'params': {'radius': 17, 'sigma_color':30, 'sigma_xy': 60}
		},
	]
})

program.append({
	'name':'dysonspherel',
	'iterations':30,
	'step_size':2.0,
	'octaves':5,
	'octave_cutoff':4,
	'octave_scale':1.6,
	'iteration_mult':0.0,
	'step_mult':0.01,
	'model':'places',
	'layers':[
		'inception_5a/5x5_reduce',
	],
	'features':[8],
	'cyclefx':[
		{
			'name': 'octave_scaler',
			'params': {'step':0.1, 'min_scale':1.5, 'max_scale':1.7}
		},
		inception_xform_default
	],
	'stepfx':[
		# {
		# 	'name': 'nd_gaussian',`
		# 	'params': {'sigma': 0.4, 'order':0}
		# },
		{
		'name': 'bilateral_filter',
		'params': {'radius': 7, 'sigma_color':30, 'sigma_xy': 60}
		},
		{
			'name': 'duration_cutoff',
			'params': {'duration':10.0}
		},
		{
			'name': 'step_opacity',
			'params': {'opacity':0.5}
		}

	]
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
		'inception_4b/3x3_reduce'
	],
	'features':[-1],
	'cyclefx':[
		{
			'name': 'octave_scaler',
			'params': {'step':0.1, 'min_scale':1.4, 'max_scale':1.7}
		},
		inception_xform_default
	],
	'stepfx':[
		# {
		# 	'name': 'nd_gaussian',
		# 	'params': {'sigma': 0.4, 'order':0}
		# },
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
	'name':'Pelicane',
	'iterations':20,
	'step_size':2,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.5,
	'iteration_mult':0.0,
	'step_mult':0.01,
	'model':'googlenet',
	'layers':[
		'inception_5b/1x1',
	],
	'features':[55],
	'cyclefx':[
		{
			'name': 'octave_scaler',
			'params': {'step':0.1, 'min_scale':1.4, 'max_scale':1.7}
		},
		inception_xform_default
	],
	'stepfx':[
		# {
		# 	'name': 'nd_gaussian',
		# 	'params': {'sigma': 0.4, 'order':0}
		# },
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
	'name':'superglam',
	'iterations':10,
	'step_size':2,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.5,
	'iteration_mult':0.0,
	'step_mult':0.02,
	'model':'googlenet',
	'layers':[
		'inception_4c/1x1',
	],
	'features':[88],
	'cyclefx':[
		{
			'name': 'octave_scaler',
			'params': {'step':0.1, 'min_scale':1.4, 'max_scale':1.7}
		},
		inception_xform_default
	],
	'stepfx':[
		# {
		# 	'name': 'nd_gaussian',
		# 	'params': {'sigma': 0.4, 'order':0}
		# },
		{
		'name': 'bilateral_filter',
		'params': {'radius': 7, 'sigma_color':30, 'sigma_xy': 60}
		},
		{
			'name': 'duration_cutoff',
			'params': {'duration':10.0}
		},
		median_blur_default
	]
})

program.append({
	'name':'Pelorat',
	'iterations':20,
	'step_size':2,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.5,
	'iteration_mult':0.0,
	'step_mult':0.01,
	'model':'googlenet',
	'layers':[
		'inception_4c/1x1',
	],
	'features':[105],
	'cyclefx':[
		{
			'name': 'octave_scaler',
			'params': {'step':0.1, 'min_scale':1.4, 'max_scale':1.7}
		},
		inception_xform_default
	],
	'stepfx':[
		# {
		# 	'name': 'nd_gaussian',
		# 	'params': {'sigma': 0.4, 'order':0}
		# },
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
	'name':'Sailing Into The West',
	'iterations':20,
	'step_size':2,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.7,
	'iteration_mult':0.0,
	'step_mult':0.01,
	'model':'googlenet',
	'layers':[
		'inception_4c/1x1'
	],
	'features':[10],
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
	'name':'Gnomicon',
	'iterations':20,
	'step_size':2,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.7,
	'iteration_mult':0.0,
	'step_mult':0.01,
	'model':'googlenet',
	'layers':[
		'inception_4d/5x5_reduce'
	],
	'features':[31],
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


