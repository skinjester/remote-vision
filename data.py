capture_size = [1920,1080]
viewport_size = [1920,1080]
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


models = {}
models['path'] = 'E:/Users/Gary/Documents/code/models'
models['cars'] = ('cars','deploy.prototxt','googlenet_finetune_web_car_iter_10000.caffemodel')
models['googlenet'] = ('bvlc_googlenet','deploy.prototxt','bvlc_googlenet.caffemodel')
models['places'] = ('googlenet_places205','deploy.prototxt','places205_train_iter_2400000.caffemodel')


layers = [
	'inception_4d/output',
	'inception_4b/5x5',
	'inception_4e/3x3',
	'inception_4d/5x5',
	'inception_4b/pool_proj',
	'inception_4c/pool_proj',
	'inception_4e_pool_proj',
	'inception_4e/pool_proj',
	'inception_4d/3x3',
	'inception_3a/1x1',
	'inception_4a/pool',
	'inception_5a/output',
	'inception_4a/3x3',
	'inception_3b/5x5',
	'conv2/norm2',
	'inception_3a/output',
	'inception_3b/3x3',
	'inception_3b/5x5_reduce',
]


settings = {}


settings['default2'] = {
	'iterations':20,
	'step_size':2,
	'octaves':6,
	'octave_cutoff':4,
	'octave_scale':1.4,
	'iteration_mult':0.5,
	'step_mult':0,
	'duration':10,
	'viewport_size':[1920,1080],
	'capture_size':[1920,1080],
	'model':'googlenet',
	'layers':[
		'inception_4a/pool',
		'inception_4d/pool'
	],
	'guides':[
		'eagle1.jpg',
		'eyeballs.jpg'
	],
	'threshold':50000
}

settings['niceplaces'] = {
	'iterations':30,
	'step_size':2,
	'octaves':6,
	'octave_cutoff':7,
	'octave_scale':1.4,
	'iteration_mult':0.5,
	'step_mult':0.01,
	'duration':8,
	'comments':'hello',
	'viewport_size':[1920,1080],
	'capture_size':[1280,720],
	'layers':[
		'inception_4a/pool',
		'inception_4d/pool'
	],
	'guides':[
		'eagle1.jpg',
		'eyeballs.jpg'
	],
	'threshold':50000
}


settings['niceplaces-good'] = {
	'iterations':20,
	'step_size':3,
	'octaves':7,
	'octave_cutoff':5,
	'octave_scale':1.5,
	'iteration_mult':0,
	'step_mult':0.0,
	'duration':276,
	'comments':'amazing. works great with places model too',
	'viewport_size':[1920,1080],
	'capture_size':[1280,720],
	'layers':[
		'inception_4a/pool',
		'inception_4d/pool'
	],
	'guides':[
		'eagle1.jpg',
		'eyeballs.jpg'
	],
	'threshold':50000
}

settings['hirez-fast'] = {
	'iterations':10,
	'step_size':3,
	'octaves':6,
	'octave_cutoff':4,
	'octave_scale':1.4,
	'iteration_mult':0,
	'step_mult':-0.00,
	'duration':58,
	'comments':'amazing. works great with places model too',
	'viewport_size':[1920,1080],
	'capture_size':[1280,720],
	'layers':[
		'inception_4a/pool',
		'inception_4d/pool'
	],
	'guides':[
		'eagle1.jpg',
		'eyeballs.jpg'
	],
	'threshold':50000
}

settings['hirez'] = {
	'iterations':30,
	'step_size':3,
	'octaves':6,
	'octave_cutoff':5,
	'octave_scale':1.4,
	'iteration_mult':0,
	'step_mult':-0.01,
	'duration':58,
	'comments':'amazing. works great with places model too',
	'viewport_size':[1920,1080],
	'capture_size':[1280,720],
	'layers':[
		'inception_4a/pool',
		'inception_4d/pool'
	],
	'guides':[
		'eagle1.jpg',
		'eyeballs.jpg'
	],
	'threshold':50000
}

settings['hirez(places)'] = {
	'iterations':100,
	'step_size':2,
	'octaves':7,
	'octave_cutoff':6,
	'octave_scale':1.6,
	'iteration_mult':0,
	'step_mult':-0.001,
	'duration':276,
	'comments':'amazing. works great with places model too',
	'viewport_size':[1920,1080],
	'capture_size':[1280,720],
	'layers':[
		'inception_4a/pool',
		'inception_4d/pool'
	],
	'guides':[
		'eagle1.jpg',
		'eyeballs.jpg'
	],
	'threshold':50000
}

settings['doingitwrong'] = {
	'iterations':20,
	'step_size':1,
	'octaves':6,
	'octave_cutoff':3,
	'octave_scale':1.4,
	'iteration_mult':0.25,
	'step_mult':0.5,
	'duration':18,
	'comments':'although overloaded, this shows how image detail is painted in'
}




settings['hifi'] = {
	'iterations':15,
	'step_size':3.0,
	'octaves':5,
	'octave_cutoff':4,
	'octave_scale':1.4,
	'iteration_mult':0.5,
	'step_mult':0.01,
	'duration':13,
	'comments':'?',
	'viewport_size':[1920,1080],
	'capture_size':[1920,1080],
	'model':'googlenet',
	'layers':[
		'inception_4a/pool',
		'inception_4d/pool'
	],
	'guides':[
		'eagle1.jpg',
		'eyeballs.jpg'
	],
	'threshold':50000
}

settings['hifi-best'] = {
	'iterations':20,
	'step_size':3.0,
	'octaves':6,
	'octave_cutoff':5,
	'octave_scale':1.4,
	'iteration_mult':0.25,
	'step_mult':0.0,
	'duration':31,
	'comments':'delivered beautiful output in low light at 1080p',
	'viewport_size':[1920,1080],
	'capture_size':[1280,720],
	'model':'googlenet',
	'layers':[
		'inception_4a/pool',
		'inception_4d/pool'
	],
	'guides':[
		'eagle1.jpg',
		'eyeballs.jpg'
	],
	'threshold':50000
}

