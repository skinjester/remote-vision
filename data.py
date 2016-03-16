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
guides.append('./img/rabbit2.jpg')
guides.append('./img/eyeballs.jpg')
guides.append('./img/spectra.jpg')
guides.append('./img/rose1.jpg')
guides.append('./img/bono.jpg')
guides.append('./img/kitteh.jpg')
guides.append('./img/kitteh2.jpg')
guides.append('./img/kitteh3.jpg')
guides.append('./img/dune.jpg')
guides.append('./img/sax2.jpg')
guides.append('./img/ghost.jpg')

models = {}
models['path'] = 'E:/Users/Gary/Documents/code/models'
models['cars'] = ('cars','deploy.prototxt','googlenet_finetune_web_car_iter_10000.caffemodel')
models['googlenet'] = ('bvlc_googlenet','deploy.prototxt','bvlc_googlenet.caffemodel')
models['places'] = ('googlenet_places205','deploy.prototxt','places205_train_iter_2400000.caffemodel')


layers = [
	'inception_4e/pool_proj',
	'inception_4d/3x3',
	'inception_4d/output',
	'inception_3a/1x1',
	'inception_4a/pool',
	'inception_4b/pool',
	'inception_5a/output',
	'inception_4c/5x5',
	'inception_3b/3x3_reduce',
	'inception_4e/output',
	'inception_4a/3x3',
	'inception_3b/5x5',
	'conv2/norm2',
	'inception_3a/output',
	'inception_3b/3x3',
	'inception_3b/5x5_reduce',
]


settings = {}
settings['default'] = {
	'iterations':40,
	'step_size':0.8,
	'octaves':6,
	'octave_cutoff':5,
	'octave_scale':1.4,
	'iteration_mult':0.5,
	'step_mult':0.02,
	'duration':11,
}

settings['default2'] = {
	'iterations':20,
	'step_size':2,
	'octaves':6,
	'octave_cutoff':4,
	'octave_scale':1.4,
	'iteration_mult':0.5,
	'step_mult':0,
	'duration':10,
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
	'step_size':6,
	'octaves':6,
	'octave_cutoff':5,
	'octave_scale':1.4,
	'iteration_mult':0.5,
	'step_mult':-0.05,
	'duration':11,
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

settings['edutainment'] = {
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

settings['niceplaces-clean'] = {
	'iterations':80,
	'step_size':0.5,
	'octaves':6,
	'octave_cutoff':4,
	'octave_scale':1.4,
	'iteration_mult':0.5,
	'step_mult':0.01,
	'duration':18,
	'comments':''
}

settings['niceplaces1'] = {
	'iterations':30,
	'step_size':3.5,
	'octaves':6,
	'octave_cutoff':7,
	'octave_scale':1.4,
	'iteration_mult':0.5,
	'step_mult':-0.001,
	'duration':10,
	'comments':''
}

settings['niceplaces-vfast'] = {
	'iterations':60,
	'step_size':3.5,
	'octaves':6,
	'octave_cutoff':3,
	'octave_scale':1.4,
	'iteration_mult':0.25,
	'step_mult':-0.001,
	'duration':10,
	'comments':''
}

settings['niceplaces-fast'] = {
	'iterations':30,
	'step_size':1,
	'octaves':7,
	'octave_cutoff':5,
	'octave_scale':1.5,
	'iteration_mult':0.75,
	'step_mult':0.05,
	'duration':5,
}

settings['mckenna'] = {
	'iterations':20,
	'step_size':1.0,
	'octaves':6,
	'octave_cutoff':4,
	'octave_scale':1.5,
	'iteration_mult':0.5,
	'step_mult':0.01,
	'duration':7,
}

settings['fast'] = {
	'iterations':50,
	'step_size':1.0,
	'octaves':6,
	'octave_cutoff':5,
	'octave_scale':1.4,
	'iteration_mult':0.5,
	'step_mult':0.01,
	'duration':14,
}
settings['fast-tighter'] = {
	'iterations':20,
	'step_size':2.0,
	'octaves':6,
	'octave_cutoff':6,
	'octave_scale':1.4,
	'iteration_mult':0.5,
	'step_mult':0.01,
	'duration':6,
}
settings['fast-d'] = {
	'iterations':20,
	'step_size':2.0,
	'octaves':6,
	'octave_cutoff':4,
	'octave_scale':1.4,
	'iteration_mult':0.5,
	'step_mult':0.01,
	'duration':5,
}
settings['fast-e'] = {
	'iterations':10,
	'step_size':2.0,
	'octaves':6,
	'octave_cutoff':4,
	'octave_scale':1.4,
	'iteration_mult':0.8,
	'step_mult':0.01,
	'duration':2,
}
settings['fast-f'] = {
	'iterations':80,
	'step_size':1.0,
	'octaves':6,
	'octave_cutoff':4,
	'octave_scale':1.4,
	'iteration_mult':0.75,
	'step_mult':0.01,
	'duration':5,
}


settings['faster'] = {
	'iterations':10,
	'step_size':4.0,
	'octaves':5,
	'octave_cutoff':4,
	'octave_scale':1.4,
	'iteration_mult':0.75,
	'step_mult':0.05,
	'duration':12,
	'comments':'?',
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

settings['hifi'] = {
	'iterations':15,
	'step_size':3.0,
	'octaves':5,
	'octave_cutoff':4,
	'octave_scale':1.4,
	'iteration_mult':0.5,
	'step_mult':0.01,
	'duration':12,
	'comments':'?',
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




settings['hifi-broad'] = {
	'iterations':20,
	'step_size':3.0,
	'octaves':6,
	'octave_cutoff':5,
	'octave_scale':1.4,
	'iteration_mult':0.25,
	'step_mult':0.0,
	'duration':31,
	'comments':'delivered beautiful output in low light at 1080p'
}

