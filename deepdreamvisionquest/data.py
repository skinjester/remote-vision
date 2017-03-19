capture_size = [1280,720]
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

# a list of programs
program = []


program.append({
	'name':'geo',
	'iterations':10,
	'step_size':6.0,
	'octaves':6,
	'octave_cutoff':3,
	'octave_scale':1.4,
	'iteration_mult':0.0,
	'step_mult':0.0,
	'model':'places',
	'layers':[
		'inception_3b/5x5',
	],
	'features':[-1,0,1]
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
	'features':range(-1,96)
})



program.append({
	'name':'geo',
	'iterations':40,
	'step_size':4.0,
	'octaves':8,
	'octave_cutoff':8,
	'octave_scale':1.5,
	'iteration_mult':0.0,
	'step_mult':0.02,
	'model':'places',
	'layers':[
		'inception_4c/output'
	],
	'features':[1]
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
	'features':range(-1,96)
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
	'features':range(-1,96)
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
	'features':range(-1,96)
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
	'features':[27]
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
	'features':[21]
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
	'features':[21]
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
	'features':[24]
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
	'features':[15]
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
	'features':[12]
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
	'features':[11]
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
	'features':[11]
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
	'features':[15]
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
	'features':[17]
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
	'features':[2]
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
	'features':[11]
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
	'features':[1]
})

# print '*' * 8
# #print program['ghost']['layers'][0]

# print program[0]['name']


# print '*' * 8
# quit()

