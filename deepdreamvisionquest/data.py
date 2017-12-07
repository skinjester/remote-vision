import numpy as np
import cv2

# processing resolution
capture_w = 1280
capture_h = 720

# capture_w = 1920
# capture_h = 1080

# capture_w = 960
# capture_h = 720

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
	'places': ('googlenet_places205','deploy.prototxt','places205_train_iter_2400000.caffemodel'),
	'vgg19': ('VGG_ILSVRC_19','deploy.prototxt','VGG_ILSVRC_19_layers.caffemodel')
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


vgg19_layers = [
	'conv3_1',
	'conv3_2',
	'conv3_3',
	'conv3_4',
	'conv4_1',
	'conv4_2',
	'conv4_3',
	'conv4_4',
	'conv5_1',
	'conv5_2',
	'conv5_3',
	'conv5_4'
]


# a list of programs
program = []

# defaults provided as a convenience
xform_array_default = {
	'name': 'xform_array',
	'params': {'amplitude':20, 'wavelength':50}
}

octave_scaler_default = {
	'name': 'octave_scaler',
	'params': {'step':0.01, 'min_scale':1.6, 'max_scale':1.7}
}

inception_xform_default = {
	'name': 'inception_xform',
	'params': {'scale':0.1}
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

# name: program name
# iterations: number of iterations per octave
# step size:

program.append({
  'name':'peyoteworld',
  'iterations':4,
  'step_size':2,
  'octaves':5,
  'octave_cutoff':4,
  'octave_scale':1.2,
  'iteration_mult':0.5,
  'step_mult':0.1,
  'model':'vgg19',
  'layers':[
	'conv3_1',
	'conv3_2',
	'conv3_3',
	'conv3_4',
	'conv4_1',
	'conv4_2',
	'conv4_3',
	'conv4_4',
	'conv5_1',
	'conv5_2',
	'conv5_3',
	'conv5_4'
	],
  'features':range(-1,255),
  'cyclefx':[
    inception_xform_default,
    {
    	'name': 'octave_scaler',
    	'params': {'step':0.1, 'min_scale':1.1, 'max_scale':1.5}
    }
  ],
  'stepfx':[

  ]
})

program.append({
	'name':'wildlife-cambrian-1',
	'iterations':10,
	'step_size':3,
	'octaves':5,
	'octave_cutoff':4,
	'octave_scale':1.5,
	'iteration_mult':0.0,
	'step_mult':-0.1,
	'model':'googlenet',
	'layers':[
		'inception_4c/pool',
	],
	'features':range(-1,256),
	'cyclefx': [
	    inception_xform_default,
	],
	'stepfx': [

	]
})

program.append({
	'name':'strangerthing',
	'iterations':5,
	'step_size':2.4,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.4,
	'iteration_mult':-0.25,
	'step_mult':0.02,
	'model':'vgg19',
	'layers':[
		'conv4_4'
	],
	'features':range(177,512),
	'cyclefx': [
	    inception_xform_default,
	    {
	    	'name': 'octave_scaler',
	    	'params': {'step':0.1, 'min_scale':1.4, 'max_scale':1.5}
	    }
	],
	'stepfx': [
		{
			'name': 'bilateral_filter',
			'params': {'radius': 3, 'sigma_color':20, 'sigma_xy': 70}
		},
	]
})

program.append({
	'name':'wildlife',
	'iterations':40,
	'step_size':3,
	'octaves':6,
	'octave_cutoff':6,
	'octave_scale':1.2,
	'iteration_mult':0.25,
	'step_mult':-0.05,
	'model':'vgg19',
	'layers':[
		'conv5_3',
	],
	'features':range(136,256),
	'cyclefx': [
		inception_xform_default,
		{
			'name': 'octave_scaler',
			'params': {'step':0.1, 'min_scale':1.1, 'max_scale':1.5}
		}
	],
	'stepfx': [
		{
			'name': 'bilateral_filter',
			'params': {'radius': 3, 'sigma_color':50, 'sigma_xy': 10}
		},
	]
})


program.append({
	'name':'violaceous',
	'iterations':20,
	'step_size':2,
	'octaves':7,
	'octave_cutoff':5,
	'octave_scale':1.4,
	'iteration_mult':0.2,
	'step_mult':0.01,
	'model':'vgg19',
	'layers':[
		'conv5_3'
	],
	'features':range(104,256),
	'cyclefx':[
			{
				'name': 'octave_scaler',
				'params': {'step':0.1, 'min_scale':1.2, 'max_scale':1.6}
			}
		],
	'stepfx': [
		{
			'name': 'bilateral_filter',
			'params': {'radius': 7, 'sigma_color':50, 'sigma_xy': 3}
		},
		{
			'name': 'nd_gaussian',
			'params': {'sigma': 0.2, 'order':0}
		},

	]
})

program.append({
	'name':'riviera',
	'iterations':20,
	'step_size':1.6,
	'octaves':6,
	'octave_cutoff':4,
	'octave_scale':1.5,
	'iteration_mult':0.0,
	'step_mult':0.001,
	'model':'places',
	'layers':[
		'inception_4c/output',
	],
	'features':range(33,100),
	'cyclefx':[
		{
			'name': 'octave_scaler',
			'params': {'step':0.01, 'min_scale':1.4, 'max_scale':1.7}
		},
		{
			'name': 'inception_xform',
			'params': {'scale':0.1}
		}
	],
	'stepfx':[

		{
		'name': 'bilateral_filter',
		'params': {'radius': 7, 'sigma_color':5, 'sigma_xy': 0}
		}
	]
})


program.append({
	'name':'cafe',
	'iterations':20,
	'step_size':1.6,
	'octaves':6,
	'octave_cutoff':4,
	'octave_scale':1.5,
	'iteration_mult':0.0,
	'step_mult':0.001,
	'model':'places',
	'layers':[
		'inception_4d/5x5',
		'inception_4c/output',
		'inception_4c/pool',
		'inception_4d/3x3'
	],
	'features':range(5,100),
	'cyclefx':[
		# {
		# 	'name': 'octave_scaler',
		# 	'params': {'step':0.1, 'min_scale':1.4, 'max_scale':1.7}
		# },
		{
			'name': 'inception_xform',
			'params': {'scale':0.1}
		}
	],
	'stepfx':[

		{
		'name': 'bilateral_filter',
		'params': {'radius': 3, 'sigma_color':10, 'sigma_xy': 60}
		}
	]
})


program.append({
	'name':'neomorph-neo-2',
	'iterations':40,
	'step_size':2,
	'octaves':5,
	'octave_cutoff':3,
	'octave_scale':1.5,
	'iteration_mult':0.0,
	'step_mult':0.01,
	'model':'googlenet',
	'layers':[
		'inception_4c/5x5'
	],
	'features':[7],
	'cyclefx':[
		{
			'name': 'octave_scaler',
			'params': {'step':0.01, 'min_scale':1.4, 'max_scale':1.7}
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
		'params': {'radius': 7, 'sigma_color':16, 'sigma_xy': 60}
		}
	]
})

program.append({
	'name':'neomorph-neo',
	'iterations':40,
	'step_size':2,
	'octaves':5,
	'octave_cutoff':3,
	'octave_scale':1.5,
	'iteration_mult':0.0,
	'step_mult':0.01,
	'model':'googlenet',
	'layers':[
		'inception_4c/5x5',
		'inception_4c/5x5_reduce',
		'inception_4c/output',
		'inception_4c/pool',
		'inception_4d/3x3',
		'inception_4d/5x5'
	],
	'features':range(64),
	'cyclefx':[
		{
			'name': 'octave_scaler',
			'params': {'step':0.01, 'min_scale':1.4, 'max_scale':1.7}
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
		'params': {'radius': 7, 'sigma_color':16, 'sigma_xy': 60}
		}
	]
})

program.append({
	'name':'neomorph',
	'iterations':20,
	'step_size':2,
	'octaves':5,
	'octave_cutoff':3,
	'octave_scale':1.5,
	'iteration_mult':0.0,
	'step_mult':0.01,
	'model':'googlenet',
	'layers':[
		'inception_4d/3x3',
		'inception_4d/5x5'
	],
	'features':range(27,128),
	'cyclefx':[
		{
			'name': 'octave_scaler',
			'params': {'step':0.05, 'min_scale':1.3, 'max_scale':1.6}
		},
		inception_xform_default
	],
	'stepfx':[
		{
		'name': 'bilateral_filter',
		'params': {'radius': 5, 'sigma_color':64, 'sigma_xy': 60}
		}
	]
})


program.append({
	'name':'geo',
	'iterations':10,
	'step_size':2.2,
	'octaves':4,
	'octave_cutoff':3,
	'octave_scale':1.4,
	'iteration_mult':0.0,
	'step_mult':0.04,
	'model':'googlenet',
	'layers':[
		'inception_3b/5x5',
	],
	'features':range(33,64),
	'cyclefx':[
		{
			'name': 'xform_array',
			'params': {'amplitude':2, 'wavelength':100}
		},
	],
	'stepfx':[
		{
			'name': 'duration_cutoff',
			'params': {'duration':2.0}
		},
		{
			'name': 'bilateral_filter',
			'params': {'radius': 3, 'sigma_color':64, 'sigma_xy': 60}
		},
	]
})





program.append({
	'name':'alien-human hybrid',
	'iterations':40,
	'step_size':2,
	'octaves':5,
	'octave_cutoff':3,
	'octave_scale':1.5,
	'iteration_mult':0.0,
	'step_mult':0.01,
	'model':'places',
	'layers':[
		'inception_4a/1x1'
	],
	'features':[0],
	'cyclefx':[
		{
			'name': 'octave_scaler',
			'params': {'step':0.05, 'min_scale':1.1, 'max_scale':1.7}
		},
		inception_xform_default
	],
	'stepfx':[
		{
			'name': 'nd_gaussian',
			'params': {'sigma': 0.3, 'order':0}
		},
		{
		'name': 'bilateral_filter',
		'params': {'radius': 7, 'sigma_color':16, 'sigma_xy': 60}
		}
	]
})




program.append({
	'name':'chainmail',
	'iterations':40,
	'step_size':2,
	'octaves':5,
	'octave_cutoff':3,
	'octave_scale':1.5,
	'iteration_mult':0.0,
	'step_mult':0.01,
	'model':'places',
	'layers':[
		'conv2/3x3'
	],
	'features':[6],
	'cyclefx':[
		{
			'name': 'octave_scaler',
			'params': {'step':0.05, 'min_scale':1.1, 'max_scale':1.7}
		},
		inception_xform_default
	],
	'stepfx':[
		{
			'name': 'nd_gaussian',
			'params': {'sigma': 0.3, 'order':0}
		},
		{
		'name': 'bilateral_filter',
		'params': {'radius': 7, 'sigma_color':16, 'sigma_xy': 60}
		},
		{
			'name': 'duration_cutoff',
			'params': {'duration':2.0}
		}
	]
})

program.append({
	'name':'sheldrake',
	'iterations':40,
	'step_size':3,
	'octaves':5,
	'octave_cutoff':4,
	'octave_scale':1.5,
	'iteration_mult':0.0,
	'step_mult':-0.02,
	'model':'googlenet',
	'layers':[
		'inception_4a/3x3',
	],
	'features':range(18,32),
	'cyclefx':[
		{
			'name': 'octave_scaler',
			'params': {'step':0.05, 'min_scale':1.1, 'max_scale':1.7}
		},
		inception_xform_default
	],
	'stepfx':[
		{
			'name': 'nd_gaussian',
			'params': {'sigma': 0.3, 'order':0}
		},
		{
		'name': 'bilateral_filter',
		'params': {'radius': 7, 'sigma_color':16, 'sigma_xy': 60}
		}
	]
})


program.append({
	'name':'Kobol2',
	'iterations':30,
	'step_size':3,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.4,
	'iteration_mult':0.0,
	'step_mult':0.00,
	'model':'places',
	'layers':[
		'inception_4b/3x3',
	],
	'features':range(3,64),
	'cyclefx':[
		{
			'name': 'octave_scaler',
			'params': {'step':0.1, 'min_scale':1.4, 'max_scale':1.7}
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
		'params': {'radius': 7, 'sigma_color':30, 'sigma_xy': 100}
		}
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
	'name':'therapod',
	'iterations':100,
	'step_size':1.0,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.4,
	'iteration_mult':0.5,
	'step_mult':0.01,
	'model':'googlenet',
	'layers':[
		'inception_4b/3x3_reduce',
		'inception_4d/3x3_reduce',
		'inception_4c/3x3_reduce',
	],
	'features':range(63,128),
	'cyclefx':[
		{
			'name': 'octave_scaler',
			'params': {'step':0.01, 'min_scale':1.3, 'max_scale':1.7}
		},
		inception_xform_default
	],
	'stepfx':[
		{
			'name': 'nd_gaussian',
			'params': {'sigma': 0.4, 'order':0}
		},

		{
			'name': 'duration_cutoff',
			'params': {'duration':5.0}
		},
	]
})

program.append({
	'name':'pebble beach yall',
	'iterations':100,
	'step_size':1.0,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.4,
	'iteration_mult':0.5,
	'step_mult':0.01,
	'model':'googlenet',
	'layers':[
		'inception_4b/3x3_reduce',
		'inception_4d/3x3_reduce',
		'inception_4c/3x3_reduce',
	],
	'features':range(63,128),
	'cyclefx':[
		{
			'name': 'octave_scaler',
			'params': {'step':0.01, 'min_scale':1.3, 'max_scale':1.7}
		},
		inception_xform_default
	],
	'stepfx':[
		{
			'name': 'nd_gaussian',
			'params': {'sigma': 0.4, 'order':0}
		},

		{
			'name': 'duration_cutoff',
			'params': {'duration':5.0}
		},
	]
})





program.append({
	'name':'hifi-featuremap',
	'iterations':5,
	'step_size':2,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.4,
	'iteration_mult':0.0,
	'step_mult':0.01,
	'model':'places',
	'layers':[
		'inception_4b/3x3_reduce',
		'inception_4c/3x3_reduce'
	],
	'features':range(54,100),
	'cyclefx':[
		{
			'name': 'octave_scaler',
			'params': {'step':0.1, 'min_scale':1.3, 'max_scale':1.7}
		},
	],
	'stepfx':[
		# {
		# 	'name': 'nd_gaussian',
		# 	'params': {'sigma': 0.4, 'order':0}
		# },
		{
		'name': 'bilateral_filter',
		'params': {'radius': 7, 'sigma_color':30, 'sigma_xy': 200}
		},
		{
			'name': 'duration_cutoff',
			'params': {'duration':5.0}
		},

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
	'name':'Shpongled',
	'iterations':20,
	'step_size':3,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.7,
	'iteration_mult':0.0,
	'step_mult':-0.02,
	'model':'googlenet',
	'layers':[
		'inception_4b/5x5_reduce'
	],
	'features':range(10,24),
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


