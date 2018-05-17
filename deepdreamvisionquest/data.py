import numpy as np
import cv2

# processing resolution
capture_w = 1280
capture_h = 720

# crashy
# capture_w = 1920
# capture_h = 1080

# capture_w = 960
# capture_h = 720

# 4K camera doesn't support this display size
capture_w = 864
capture_h = 480

# capture_w = 640
# capture_h = 360

# capture_w = 1280
# capture_h = 800



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
	'vgg19': ('VGG_ILSVRC_16','deploy.prototxt','VGG_ILSVRC_16_layers.caffemodel')
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
	'params': {'scale':0.075}
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
	'name':'cambrian-explanation',
	'iterations':5,
	'step_size':3.,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.5,
	'iteration_mult':-0.2,
	'step_mult':0.03,
	'model':'googlenet',
	'layers':[
		'inception_4c/pool',
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
	],
	'features':range(-1,256),
	'cyclefx': [
		# {
		# 	'name': 'inception_xform',
		# 	'params': {'scale':0.2}
		# },
		# {
		#   	'name': 'octave_scaler',
		#   	'params': {'step':0.1, 'min_scale':153, 'max_scale':2.0}
		# },
	],
	'stepfx': [
		# {
		# 	'name': 'nd_gaussian',
		# 	'params': {'sigma': 0.4, 'order':0}
		# },
		{
			'name': 'bilateral_filter',
			'params': {'radius': 3, 'sigma_color':50, 'sigma_xy': 20}
		},
		{
		  	'name': 'octave_scaler',
		  	'params': {'step':0.01, 'min_scale':1.4, 'max_scale':2.0}
		},
	]
})

program.append({
	'name':'Hypercube',
	'iterations':30,
	'step_size':1.5,
	'octaves':4,
	'octave_cutoff':4,
	'octave_scale':1.8,
	'iteration_mult':0.25,
	'step_mult':-0.001,
	'model':'places',
	'layers':[
		'inception_4c/1x1',
		'inception_4c/3x3',
	],
	'features':range(119,256),
	'cyclefx': [
		{
			'name': 'inception_xform',
			'params': {'scale':0.1}
		},
		# {
		# 	'name': 'octave_scaler',
		# 	'params': {'step':0.1, 'min_scale':1.5, 'max_scale':2.0}
		# },
	],
	'stepfx': [
		# {
		# 	'name': 'step_opacity',
		# 	'params': {'opacity':0.5}
		# },
		{
			'name': 'bilateral_filter',
			'params': {'radius': 3, 'sigma_color':5, 'sigma_xy': 10}
		},
	]
})

program.append({
	'name':'GAIA',
	'iterations':10,
	'step_size':2.5,
	'octaves':4,
	'octave_cutoff':3,
	'octave_scale':1.8,
	'iteration_mult':0.25,
	'step_mult':0.01,
	'model':'places',
	'layers':[
		'inception_4c/1x1',
		'inception_4c/3x3',
	],
	'features':range(111,256),
	'cyclefx': [
		{
			'name': 'inception_xform',
			'params': {'scale':0.1}
		},
		{
			'name': 'octave_scaler',
			'params': {'step':0.2, 'min_scale':1.1, 'max_scale':2.0}
		},
	],
	'stepfx': [
		{
			'name': 'bilateral_filter',
			'params': {'radius': 3, 'sigma_color':20, 'sigma_xy': 50}
		},

	]
})

program.append({
	'name':'cambrian-candidate-places',
	'iterations':20,
	'step_size':2,
	'octaves':4,
	'octave_cutoff':4,
	'octave_scale':1.7,
	'iteration_mult':0.0,
	'step_mult':0.0,
	'model':'places',
	'layers':[
		'inception_4b/pool',
		'inception_4c/pool',
		'inception_4b/3x3_reduce',
		'inception_4b/5x5',
		'inception_4b/5x5_reduce',
		'inception_4b/output',
		'inception_4b/pool_proj',
		'inception_4c/1x1',
		'inception_4c/3x3',
		'inception_4c/3x3_reduce',
		'inception_4c/5x5',
		'inception_4c/5x5_reduce',
		'inception_4c/output',
		'inception_3a/1x1',
		'inception_3a/3x3',
		'inception_3b/5x5',
		'inception_3b/output',
		'inception_3b/pool',
	],
	'features':range(-1,256),
	'cyclefx': [
		{
			'name': 'inception_xform',
			'params': {'scale':0.05}
		},
		{
			'name': 'octave_scaler',
			'params': {'step':0.01, 'min_scale':1.4, 'max_scale':2.2}
		},
	],
	'stepfx': [
		{
			'name': 'bilateral_filter',
			'params': {'radius': 3, 'sigma_color':20, 'sigma_xy': 15}
		},

	]
})

program.append({
	'name':'cambrian-candidate-googlenet',
	'iterations':20,
	'step_size':2,
	'octaves':4,
	'octave_cutoff':4,
	'octave_scale':1.7,
	'iteration_mult':0.0,
	'step_mult':0.0,
	'model':'googlenet',
	'layers':[
		'inception_4b/pool',
		'inception_4c/pool',
		'inception_4b/3x3_reduce',
		'inception_4b/5x5',
		'inception_4b/5x5_reduce',
		'inception_4b/output',
		'inception_4b/pool_proj',
		'inception_4c/1x1',
		'inception_4c/3x3',
		'inception_4c/3x3_reduce',
		'inception_4c/5x5',
		'inception_4c/5x5_reduce',
		'inception_4c/output',
		'inception_3a/1x1',
		'inception_3a/3x3',
		'inception_3b/5x5',
		'inception_3b/output',
		'inception_3b/pool',
	],
	'features':range(-1,256),
	'cyclefx': [
		{
			'name': 'inception_xform',
			'params': {'scale':0.05}
		},
		{
			'name': 'octave_scaler',
			'params': {'step':0.01, 'min_scale':1.4, 'max_scale':2.2}
		},
	],
	'stepfx': [
		{
			'name': 'bilateral_filter',
			'params': {'radius': 3, 'sigma_color':20, 'sigma_xy': 15}
		},

	]
})


# program.append({
#   'name':'metamachine',
#   'iterations':5,
#   'step_size':1.8,
#   'octaves':4,
#   'octave_cutoff':4,
#   'octave_scale':1.8,
#   'iteration_mult':0.0,
#   'step_mult':-0.25,
#   'model':'vgg19',
#   'layers':[
#   	  'conv5_1',
# 	  'conv3_1',
# 	  'conv3_2',
# 	  'conv3_3',
# 	  'conv3_4',
# 	  'conv4_1',
# 	  'conv4_2',
# 	  'conv4_3',
# 	  'conv4_4',
# 	  'conv5_3',
# 	],
#   'features':range(87,512),
#   'cyclefx':[
# 	    {
# 	    	'name': 'inception_xform',
# 	    	'params': {'scale':0.1}
# 	    },
#   ],
#   'stepfx':[]
# })


# crashes at 1080p
# program.append({
#   'name':'Robot Lover',
#   'iterations':10,
#   'step_size':1.2,
#   'octaves':5,
#   'octave_cutoff':5,
#   'octave_scale':1.1,
#   'iteration_mult':0.1,
#   'step_mult':0.02,
#   'model':'vgg19',
#   'layers':[
# 	  'conv3_3',
# 	  'conv3_4',
# 	  'conv3_1',
# 	  'conv3_2',
# 	],
#   'features':range(33,256),
#   'cyclefx':[
# 	    {
# 	    	'name': 'inception_xform',
# 	    	'params': {'scale':0.1}
# 	    },
# 	    {
# 	    	'name': 'octave_scaler',
# 	    	'params': {'step':0.02, 'min_scale':1.1, 'max_scale':1.5}
# 	    },
#   ],
#   'stepfx':[
# 		{
# 			'name': 'bilateral_filter',
# 			'params': {'radius': 5, 'sigma_color':20, 'sigma_xy': 10}
# 		},
#   ]
# })

# crashes at 1080p
# program.append({
#   'name':'JOI.00',
#   'iterations':40,
#   'step_size':1.2,
#   'octaves':6,
#   'octave_cutoff':5,
#   'octave_scale':1.3,
#   'iteration_mult':0.5,
#   'step_mult':0.02,
#   'model':'vgg19',
#   'layers':[
# 	  'conv3_3',
# 	  'conv3_1',
# 	  'conv3_2',
# 	  'conv3_3',
# 	  'conv3_4',
# 	  'conv4_1',
# 	  'conv4_2',
# 	  'conv4_3',
# 	  'conv4_4',
# 	  'conv5_1',
# 	  'conv5_2',
# 	  'conv5_3',
# 	],
#   'features':range(-1,256),
#   'cyclefx':[
# 	    {
# 	    	'name': 'inception_xform',
# 	    	'params': {'scale':0.1}
# 	    },
# 	    {
# 	    	'name': 'octave_scaler',
# 	    	'params': {'step':0.05, 'min_scale':1.2, 'max_scale':1.5}
# 	    },
#   ],
#   'stepfx':[
# 		{
# 			'name': 'bilateral_filter',
# 			'params': {'radius': 3, 'sigma_color':80, 'sigma_xy': 20}
# 		},
#   ]
# })



# program.append({
#   'name':'SEAWALL.01',
#   'iterations':10,
#   'step_size':2.2,
#   'octaves':6,
#   'octave_cutoff':6,
#   'octave_scale':1.7,
#   'iteration_mult':0.0,
#   'step_mult':0.05,
#   'model':'vgg19',
#   'layers':[
# 	'conv4_2'
# 	],
#   'features':[115, 33, 144, 88, 101, 114, 121],
#   'cyclefx':[
# 	    {
# 	    	'name': 'inception_xform',
# 	    	'params': {'scale':0.2}
# 	    },
# 	    {
# 	    	'name': 'octave_scaler',
# 	    	'params': {'step':0.05, 'min_scale':1.3, 'max_scale':1.7}
# 	    },

#   ],
#   'stepfx':[
# 		{
# 			'name': 'bilateral_filter',
# 			'params': {'radius': 3, 'sigma_color':50, 'sigma_xy': 100}
# 		},
# 		{
# 			'name': 'median_blur',
# 			'params': {'kernel_shape':3}
# 		},
# 		# {
# 		# 	'name': 'duration_cutoff',
# 		# 	'params': {'duration':8.0}
# 		# }
#   ]
# })

program.append({
  'name':'peyoteworld',
  'iterations':4,
  'step_size':1.5,
  'octaves':5,
  'octave_cutoff':3,
  'octave_scale':1.2,
  'iteration_mult':0.0,
  'step_mult':0.2,
  'model':'vgg19',
  'layers':[
	'conv3_1',
	'conv3_2',
	'conv3_3',
	],
  'features':range(-1,255),
  'cyclefx':[
    {
    	'name': 'inception_xform',
    	'params': {'scale':0.01}
    },
  ],
  'stepfx':[
    {
    	'name': 'octave_scaler',
    	'params': {'step':0.1, 'min_scale':1.1, 'max_scale':1.6}
    }

  ]
})


program.append({
  'name':'ACCIO',
  'iterations':10,
  'step_size':2,
  'octaves':4,
  'octave_cutoff':4,
  'octave_scale':1.7,
  'iteration_mult':0.5,
  'step_mult':0.01,
  'model':'vgg19',
  'layers':[
	'conv4_3',
	'conv3_3',
	'conv4_2',
	'conv3_1',
	'conv3_2',
	'conv3_4',
	'conv4_1',
	'conv4_4',
	'conv5_1',
	'conv5_2',
	'conv5_3',
	'conv5_4'
	],
  'features':range(34,255),
  'cyclefx':[
    {
    	'name': 'inception_xform',
    	'params': {'scale':0.025}
    },
    # {
    # 	'name': 'octave_scaler',
    # 	'params': {'step':0.1, 'min_scale':1.3, 'max_scale':1.8}
    # }
  ],
  'stepfx':[
	  {
	  	'name': 'bilateral_filter',
	  	'params': {'radius': 3, 'sigma_color':20, 'sigma_xy': 100}
	  },
  ]
})



program.append({
	'name':'cambrian-implosion',
	'iterations':10,
	'step_size':1.8,
	'octaves':5,
	'octave_cutoff':5,
	'octave_scale':1.5,
	'iteration_mult':0.25,
	'step_mult':-0.05,
	'model':'googlenet',
	'layers':[
		'inception_4c/pool',
		'inception_5a/output',
		'inception_5a/pool',
		'inception_5b/1x1',
		'inception_5b/3x3',
		'inception_5b/3x3_reduce',
	],
	'features':range(-1,256),
	'cyclefx': [
	    inception_xform_default,
	    {
	    	'name': 'octave_scaler',
	    	'params': {'step':0.1, 'min_scale':1.3, 'max_scale':1.6}
	    }
	],
	'stepfx': [

	    {
	    	'name': 'bilateral_filter',
	    	'params': {'radius': 3, 'sigma_color':10, 'sigma_xy': 10}
	    },
	]
})

# program.append({
# 	'name':'wildlife',
# 	'iterations':10,
# 	'step_size':1.4,
# 	'octaves':6,
# 	'octave_cutoff':6,
# 	'octave_scale':1.2,
# 	'iteration_mult':0.0,
# 	'step_mult':0.05,
# 	'model':'vgg19',
# 	'layers':[
# 		'conv5_3'
# 	],
# 	'features':range(104,256),
# 	'cyclefx':[
# 			{
# 				'name': 'octave_scaler',
# 				'params': {'step':0.01, 'min_scale':1.2, 'max_scale':1.6}
# 			}
# 		],
# 	'stepfx': [

# 		# {
# 		# 	'name': 'nd_gaussian',
# 		# 	'params': {'sigma': 0.2, 'order':0}
# 		# },
# 		{
# 			'name': 'bilateral_filter',
# 			'params': {'radius': 3, 'sigma_color':60, 'sigma_xy': 100}
# 		},

# 	]
# })

program.append({
	'name':'monaco',
	'iterations':30,
	'step_size':1.6,
	'octaves':5,
	'octave_cutoff':4,
	'octave_scale':1.5,
	'iteration_mult':0.0,
	'step_mult':0.005,
	'model':'places',
	'layers':[
		'inception_4c/output',
	],
	'features':range(39,100),
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
	'iteration_mult':-0.2,
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
		{
			'name': 'inception_xform',
			'params': {'scale':0.1}
		}
	],
	'stepfx':[

		{
			'name': 'octave_scaler',
			'params': {'step':0.01, 'min_scale':1.4, 'max_scale':1.7}
		},
		{
			'name': 'bilateral_filter',
			'params': {'radius': 3, 'sigma_color':10, 'sigma_xy': 60}
		}
	]
})




program.append({
	'name':'neomorph-neo',
	'iterations':10,
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
		inception_xform_default
	],
	'stepfx':[
		{
			'name': 'octave_scaler',
			'params': {'step':0.01, 'min_scale':1.4, 'max_scale':1.7}
		},
		# {
		# 	'name': 'nd_gaussian',
		# 	'params': {'sigma': 0.4, 'order':0}
		# },
		{
		'name': 'bilateral_filter',
		'params': {'radius': 7, 'sigma_color':16, 'sigma_xy': 60}
		}
	]
})

# program.append({
# 	'name':'neomorph',
# 	'iterations':20,
# 	'step_size':2,
# 	'octaves':5,
# 	'octave_cutoff':4,
# 	'octave_scale':1.5,
# 	'iteration_mult':0.0,
# 	'step_mult':0.01,
# 	'model':'googlenet',
# 	'layers':[
# 		'inception_4d/3x3',
# 		'inception_4d/5x5'
# 	],
# 	'features':range(27,128),
# 	'cyclefx':[
# 		{
# 			'name': 'octave_scaler',
# 			'params': {'step':0.05, 'min_scale':1.3, 'max_scale':1.6}
# 		},
# 		inception_xform_default
# 	],
# 	'stepfx':[
# 		{
# 		'name': 'bilateral_filter',
# 		'params': {'radius': 5, 'sigma_color':30, 'sigma_xy': 60}
# 		}
# 	]
# })


