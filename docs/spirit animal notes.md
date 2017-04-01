2016-01-19 00:11:32
studying the deep dream code. have been doing so fairly seriously for the past few days.
reacquainting myself with the environment. understanding it better

net.blobs['data']  	# input image is stored (caffe format) in network's data blob
net.blobs[end]		# destination layer (end = layer name)
src					# input image (instance of net.blobs['data'])
src.data
src.data[:]
src.data[0]			# image data apparently stored here?
src.diff[0]			# back propagated error correction matrix (2D)
detail				# an image array created to draw network produced detail
octave_base			# the image contained in the octave currently computed
dst					# the neural layer identified as the end layer
dst.diff[:]			# assumed to be the 2D neural weights & structure of the end layer (?)
dst.data
octave_base			# the current image stage (octave) being dreamed upon



images in OpenCV are Numpy arrays



blob (binary large object) often used to multimedia data types such as images and audio

thinking about arrays more clearly:

for an array with shape (2,2,3): 	2 layers of (2,3) 				= 12 items
for an array with shape (2,3): 		2 layers of a list of 3 items 	= 6 items
for an array with shape (3):		a list of 3 items 				= 3 items

reshape method allows the number of dimensions and the size of each dimension to be changed as long as the total number of array elements remains the same



2016-01-25 22:20:07
what happens if you run rem.py right now?

2016-07-26 07:24:20
apparently I was self documenting here, but it never really took off. Picking it back up as I tyake a look at what the v1 project needs to be

2016-07-26 07:24:50
At the moment I'm going thru the code line by line cleaning up the formatting so the Linter program (Adaconbda/Sublime Text3) stops complaining. Busy work? Maybe - but is a good way to get a sense of what I had previously written

2016-07-26 07:44:04
Or maybe not - tedious. I'll make sure everything is clean

2016-07-27 07:41:33
Getting started mapping out the signal flowe of this program. Its more complex (and likely redundant) than I know how to move forward with

2016-09-06 21:33:30
I lost my way, thought I found it again, sort of did, athen lost it again. But I'm here now. I'm back.
Finishing the website turned out to be more than just taking some amateur video in my living room. It was nearly 2 months ago, right before going to nucl.ai conference. I had so much fun that night and that morning playing with the "AI"

Watching it in the living room (as a video) isn't compelling. I have an idea about doing a shoot with models or dancers and projecting the imagery. A sort of beta test as visual support for live performance. What would it take to make that happen. hire models? book a space?
A good intermediate ide ais to start showing the project off socially - because its easier to meet people when theyre aliens too

Another intermediate is to record output directly from the display. I think I'd done so with the microsoft game recorder previously?

Some new developments:
1. I spent time working with RPyC which allows for asynchronous callbacks that let one script communicate with another. My intention is to decouple the motion detection component from the game loop and run it as a seperate process. I'd played with Visual Studio to profile the code. Unsurprisingly, most of the time is spent processing the NN in Caffe. My expectation is that motion detection can be made to work more fluidly.

2. nVidia has newer faster GPU's. Significantly faster. I wonder how much speedup could be realized running on the new TitanX?

3. Studied Pythin basics a bit further. Everywhere I used a Class, I should have used a Module

4. Ive not been able to visualize the code to my satisfgaction. Partly because I dont know what I'm doing, partly because I haven't defined the problem clearly enough. What do you hope to gain from the desired output.

5. Rebranding. I like the name. But its a mouthful. I've noty come up with anything more compelling. When the project was named, my understanding was much vaguer. nonexistent really. What is my uderstanding now?

6. Need to do a site audit - tasks, priorities. There's a lot of work I can do right now.

2016-09-09 11:50:40
I really want to rename this thing. I dont know why, but cant let it go
Reading about Bestiaris.

"The bestiary, then, is also a reference to the symbolic language of animals"

2016-09-11 19:11:01
I'm submitting a proposal for CODAME Art & Tech Exhibition
ARTIFICIAL EXPERIENCES is the name of the exhibit
Proposal Due by Saturday 9/17
Artist notification by 9/21
Showing in SF 11/11


Description of Project *
A project description in 500 words (max) that includes any collaborative partners and relation to theme "Interface"

Links to high-resolution image(s) of your artist image and your proposed work
Estimated Materials Cost and Time
Artist Biography * A 100-200 word bio for publication in press materials (reference bios on CODAME site)
Artistic Resume including past works, exhibitions, commissions, videography (provide a list of videogames, etc)
Technical specifications
How large is the artwork, size on the wall or floor footprint (metric preferred)? Any technical requirements like wifi or power? How long does it take to set up and break down?



2016-09-15 14:16:04 DATA
Designed humane user interfaces for AAA videogames enjoyed by millions internationally.

Has worked on games including Madden NFL, The Sims, Star Wars: The Force Unleashed and The Elder Scrolls Online

Creative director

On the verge of a personal vision of collaborating with artificial inn telligene as a kind of robopsychiatrist. They;'re a better breed than us, but we made them so

Invisible interfaces

Designs for your thumbs

Grateful that humans don't behave rationally. It makes the job easier

good friend
loving husband
flawed human not hopeless

Became bored with that life and is a fine artisty afgain

living link betwee the 80's and the new era of machine hallucination

networked 16 Apple IIew computers using their MIDI ports so that I could make all the computers in the lab blink at the same time. It was uncanny. 


I lost my way, thought I found it again, sort of did, athen lost it again. But I'm here now. I'm back.


Experience an interactive psychedelic journey with a computer. Using the DeepDream convolutional neural network algorithm and real-time video feedback, the system turns your image into a vision of its own thought processes--a magic mirror. Questions about DeepDream, the magic mirror setup, and the spirit realm inside the machine are all welcome.

Takeaway
Attendees will leave with an understanding of how neural networks may be used for image synthesis, and specific steps for creating their own Deep Dreaming Magic Mirror.

Intended Audience
Anyone interested in interactive art experiences will be glad they came.

Finally! Someone speaking my language! The language of Spirit.
A language that cannot be used in the workplace. A language rarely acceptable in social gatherings (including most churches). Truth be told, the so-called "Spiritual Path" has led me to place of isolation. When I saw "Find Your Spirit Animal In A Deep Dream Vision Quest" - I quietly hoped I would meet someone that I could talk to. I sometimes yearn for belonging, but I refuse to shapeshift just to fit into someone else's tribe.

When I discovered it was YOU giving the presentation, I could not control my enthusiasm! I could not wait to see you again. I realize we cannot know who we are now from the slice of childhood we briefly shared. But the moment you started talking about "masks" - I could totally picture you and I having that conversation (I wanted to have it right there and then!). Your presentation was very exciting to me. And, if I may, let me say that you have a charismatic presence on stage. But I digress...

Simply stated, I would love to hang out with you. At the very least, we should get together and catch up on over 30 years. Let me share my contact details with you.

2016-09-15 15:29:51
Anyone or anything that has influenced the artist’s artworks.

Any education or training in the field of art

Any related experience in the field of art

A summary of the artist’s artistic philosophy

Any artistic insights or techniques that are employed by the artist

2016-09-15 15:29:48
The bio should summarize the artist’s practice—including medium(s), themes, techniques, and influences

mediums
photography
videogames
sound design
performance art

themes
the alien in the familiar, the familiar in the alien
the language of the spirit
Mythology and storytelling
Emptiness


2016-09-15 15:29:45
The bio should open with a first line that encapsulates, as far as possible, what is most significant about the artist and his or her work, rather than opening with biographical tidbits, such as where the artist went to school, grew up, etc. For example: John Chamberlain is best known for his twisting sculptures made from scrap metal and banged up, discarded automobile parts and other industrial detritus.



DESCRIPTION OF PROJECT
A project description in 500 words (max) that includes any collaborative partners and relation to theme "Interface"

Experience an interactive psychedelic journey within a computer interface. Using the DeepDream convolutional neural network algorithm and real-time video feedback, the system turns your image into a vision of its own thought processes--a magic mirror. Questions about DeepDream, the magic mirror setup, and the spirit realm inside the machine are all welcome. Attendees will leave with an understanding of how neural networks may be used for image synthesis

LINKS TO HIGH-RESOLUTION IMAGE(S) OF YOUR ARTIST IMAGE AND YOUR PROPOSED WORK
artist image:
[find a picture you like]

proposed work:
http://www.deepdreamvisionquest.com/

ESTIMATED MATERIALS COST AND TIME
N/A


ARTIST BIOGRAPHY
A 100-200 word bio for publication in press materials

Gary Boodhoo combines videogames, machine learning and interface design to discover ancient images — spirit animals. Born in Jamaica, relocation to the United States provided a crash course in how to construct mythology out of the 1980's. Then computers happened and then Dungeons and Dragons happened. Today he is an industry veteran. He designs and bleeds user interfaces for videogames including The Sims and The Elder Scrolls Online. His work examines the rhythms of emergent behavior in shared digital environments. He lives in San Francisco and develops humane experiences for game studios and other creative clients.


ARTISTIC RESUME 
(including past works, exhibitions, commissions, videography (provide a list of videogames, etc)

The Ghost in the Machine Has Many Mansions, 2016
Part of a speaker series on technological rituals
Presented by the Society for Ritual Arts
http://societyforritualarts.com/join-us-on-1219-in-san-francisco-for-the-ghost-in-the-machine-has-many-mansions/

DeepDreamVisionQuest, 2016
presented for the Game Developers Conference 2016, San Francisco
http://schedule.gdconf.com/session/find-your-spirit-animal-in-a-deep-dream-vision-quest

The Elder Scrolls Online, 2015, 
A massively multiplayer online roleplaying game
platform: PlayStation 4, XBox One, Windows, OSX
publisher: Bethesda Softworks
developer: Zenimax Online Studios

Zombie Apocalypse, 2009,
An apocalyptic multiplayer shoot-em-up
platforms: PlayStation 3, XBox 360
publisher: Konami
developer: Nihilistic Software

The Sims 3, 2009, 
A life simulator game
platform:Windows, OSX
published and developed by Electronic Arts

Star Wars: The Force Unleashed, 2008, 
An epic science fiction action-adventure game
platform: PlayStation 3, XBox 360
published and developed by LucasArts

Madden NFL, 2004,
A football simulation game
platform: PlayStation 2, XBox, Windows
published and developed by Electronic Arts



TECHNICAL SPECIFICATIONS
How large is the artwork, size on the wall or floor footprint (metric preferred)? Any technical requirements like wifi or power? How long does it take to set up and break down


DeepDreamVisionQuest space requirements are flexible. It can be run sitting in front of a computer with a webcam. The technology was designed to be used socially so there must tbe room to comfortably observe and participate. There are 3 working areas to consider. Previous exhibits worked best when these areas were in close proximity to one another, with room for the computer and operator off to the side.

1. Staging area for live video capture
-	3.5 square meters
-	Hardware
	-	lighting (2)
		-	prefer to use house lighting when possible
	-	webcam
	-	tripod 
	-	chair

2. Display area
-	Requires line of sight to staging area so participants can interact with video imagery
-	Hardware
	-	Large TV or projector (depends on the space)
		-	prefer to use house systems when possible

3. Control area
-	Positioned less than 3m from Staging Area to accomodate cable runs
-	Hardware
	-	desktop computer
		-	USB input from camera
		-	HDMI output to display	
	-	computer monitor
		-	need surface to place or mount monitor
	-	computer keyboard
		-	need surface to place or mount keyboard
	-	Game controller


HARDWARE BREAKDOWN
Lighting (2) [can venue provide ?]
Large TV or Projector [can venue provide?]
Table/Desk [can venue provide?]
Chair [can venue provide?]
Webcam
Tripod
Desktop computer
Computer monitor
Computer Keyboard
Game controller
30' USB cable
HDMI cable (length TBD pending venue details)

SETUP/TEARDOWMN
Setup/Teardown takes 30 minutes. I will need to do a technical rehearsal in the space before going live



2016-09-17 01:14:44
After all, the computer trained on pictures all humans understand. We train them by showing them many examples of what we want them to learn.

2016-10-04 10:23:00
The project has moved over to Linux for scalability and so forth. I'm using a new Titan X Pascal gfx card and needed to go through a chain of dependencies to work properly - which wasn't happening in Windows. I am seeing a bit of a speedup in basic dreaming, and have confidence that seperating motion detection into a different process will also help.

There's so much to think about - what do I really want to do with this? I need to assume there is no operator other than myself if necessary - would like to see more behaviors happening on their own - or possibly in response to inputs other than the game controller?
Would it be difficult to integrate MIDI?

2016-10-04 10:28:00
Why am I unable to change the size of the frame buffer?
Oh - ut was at the top of the script
Wow! Dreaming at 960 x 540 is extremely fast, but seems like motion detection gets broken?
Oh - its the threshold values (counting fewer pixels now)

2016-10-04 10:34:00
Its so much faster that I must rethink what's happening w the motion detection and transition between dreamed frames

2016-10-04 20:50:44
studying that code. What's the plan for Artificial Experience?

2016-10-08 10:21:25
doing some houisecleaning w the deepdreamvisionquest code. I think that everything that's currently a class, shoule be a module for easier maintainability and understanding wtf is going on

-	MotionDetector
-	Viewport
-	Framebuffer
-	Model
-	Amplifier

2016-10-08 13:32:45
modules are:
-	a python file w some functions and/or variabes in it
-	you import that filee into another
-	you access the functions or variables in that module using the dot (.) operator


2016-10-08 14:28:41
still reading up on modules and remembering how the code works
One observation is that the viewport sizing I;'m doing isn't consistent
The expectation is to be rendering and processing at 960 x 540 (a rem cycle lasts 4 secons)
and scaling that view to 1920 x 1080, but is seems like that's getting mixed up and maybe its only the camera images that are coming in at the lowere rez?

2016-10-08 14:54:32
maybe the uprez can be done by the OS itself - scaling the output x2?

2016-10-08 17:10:41
dont fixate on the interaction at this time - get the structure you want in place

2016-10-08 19:00:30
To get a list of the modules that have already been imported, you can look up sys.modules.keys()





2016-10-09 00:21:22
still refactoring the motion detection class 
haven't yet converted it into a model
good improvements on responsiveness though - enough so that the idea of running it in a seperate process doesn't seem as high priority as before.
getting decent balance between speed and quality at 1280 x 720 w cycle time of 2 sec

2016-10-09 13:34:31
It isn't making sense converting from classes to modules - and for what? learning? pride? convenience?
Is a major pain in th ass to croll around this single rem.py. Would be so much easier to bve dealing with seperate files as functional chunks. 
What am I not getting?
-	global state in a module isn't being retained in functions without messy and weird global declarations for each usage in a function.
	-	A class construct just handles this shit better

- Each of the modules is totally unique. Its not like they could ever run on their own or be plugged into some other code as-is

- all i need is the ability to maintain seperate namespaces


2016-10-09 15:41:01
I'm not sure how much effort to place into restructuring this code in a more manageble way. Its not unmanageable, but can't shake the feeling that its built on an unsteady foundation, hence the desire to look further

I'm taking another look at that camera capture demo (cameo) that I'd studfies a few monbths ago

program hub
	import SomeModule
	class ComputerProgram
		__init__(self,params)
			classvars
			instatiate SomeModule.class()
		run(self)
			SomeModule.class().function()
		functions(self)
	if __name__ == '__main__':
		ComputerProgram().run()

SomeModule
	class Module
		__init__(self,params)
			classvars
		functions(self)

2016-10-09 17:54:23
I've looked into structural optimization as far as I can for the moment
Its clear that there's a real issue (opportunity?) with these various bundles of state/behavior referencing one another globally

2016-10-09 23:19:56
pushed and committed recent work to git

2016-10-10 01:28:28
I've externalized the class definition of MotionDetector

2016-10-12 09:01:57
Trying to understand what this code is doing so I can modify it
Taking Cory's advice in following what the data is doing. Very difficult to think of these little machines as small bits of functionality.

------------------------------------------
How it works, a romance in simple language
------------------------------------------

#	CAMERAS are STORED in MOTIONDETECTOR
#	CAMERA is STORED in FRAMEBUFFER1

#	GAME LOOP
		Framebuffer.is_new_cycle = True
	#	SEND FRAMEBUFFER1 to VIEWPORT
	#	PROCESS the CAMERAS STORED in the MOTIONDETECTOR
		-	STORE HISTORY to prepare for new input
			-	self.wasMotionDetected is stored as HISTORY
			-	self.delta_count is stored as HISTORY
		-	the difference between CAMERAS IS STORED as self.t_delta_framebuffer
		-	the COUNT of non-zero pixels in self.t_delta_framebuffer IS STORED as self.delta_count

		#	OUTCOMES
			-	OVERFLOW: all delta_count(s) were above detection threshold
				-	self.delta_count == 0
			-	MOTION DETECTED
				-	delta_count > delta_trigger AND HISTORY < delta_trigger
					-	SET self.wasMotionDetected = TRUE
			-	MOTION ENDED
				-	delta_count < delta_trigger and HISTORY > delta_trigger
				-	SET self.wasMotionDetected = FALSE
			-	BENEATH THRESHOLD
				-	SET self.wasMotionDetected = FALSE

		#	REFRESH the images STORED in the MOTIONDETECTOR
			-	ADD new CAMERA to QUEUE
			-	REMOVE oldest CAMERA from QUEUE
	#	FRAMEBUFFER1 = DEEPDREAM(FRAMEBUFFER,etc.)
		-	NET
		-	ITERATION_MAX
		-	OCTAVES
		-	OCTAVE_SCALE
		-	STEP_SIZE
		-	MODEL.end

	#	DEEPDREAM
		#	SETUPOCTAVES
			#	Framebuffer.is_new_cycle = FALSE
			#	INITIALIZE the OCTAVEARRAY with NETBLOB
			#	INITIALIZE the DETAIL BUFFER to STORE network output
			-	we can provide new values for OCTAVE_SCALE, ITERATION_MAX during SETUPOCTAVES

		#	OCTAVEARRAY CYCLE
			#	ITERATE from small to large
				-	OCTAVE is index for OCTAVEARRAY
				-	OCTAVE_CURRENT is framebuffer for each OCTAVE
				-	RESAMPLE DETAIL to mat ch shape of OCTAVE_CURRENT
					-	DETAIL comes from SETUPOCTAVES or previous OCTAVEARRAY CYCLE
				-	RESIZE the NETBLOB
				-	ADD DETAILS + OCTAVE_CURRENT to NETBLOB

				#	OCTAVECYCLE
					-	WHILE STEPCOUNT < ITERATION_MAX
						-	PROCESS the CAMERAS STORED in the MOTIONDETECTOR
						-	WasMotionDetected ?
							-	BREAK out of OCTAVECYCLE 

						#	MAKESTEP (model, OBJECTIVE, step parameters)
							-	parameters arrive here from initial deepdream function call
							-	makestep operates upon the neural net's data blob
							-	we can affect guide image and step size here (per iteration)
							-	NETFORWARD to OBJECTIVE
							-	NETBACKWARD
							-	APPLY gradient ascent step to NETBLOB
							#	POSTPROCESS NETBLOB
								-	dat gaussian blur
								-	

						-	CONVERT NETBLOB to RGB and WRITE to FRAMEBUFFER1
						-	SEND FRAMEBUFFER1 to Viewport
						-	OCTAVE_CURRENT ++

				#	EARLYEXIT
					#	LASTOCTAVE
						-	OCTAVEARRAY CYCLE reached OCTAVE_CUTOFF
						-	SET Framebuffer.is_dreamy = TRUE
						-	CONVERT NETBLOB to RGB and STORE as EARLY_EXIT
						-	Return EARLY_EXIT

					#	MOTION WAS DETECTED
						-	SET Framebuffer.is_dreamy = FALSE
						-	CONVERT NETBLOB to RGB and WRITE to framebuffer2
						-	Return CAMERA

				#	WRAPUP OCTAVECYCLE
					-	STORE the DETAILS produced during OCTAVECYCLE
					-

			#	WRAPUP OCTAVEARRAYCYCLE
				-	SET Framebuffer.is_dreamy = TRUE
				-	CONVERT NETBLOB to RGB and STORE as FINISHED
				-	Return FINISHED


#	FRAMEBUFFER

------------------------------
What is the COMPOSER?
	NumPy array with shape (height,width,RGB)		
	Always contains RGB image data
	Stores Game Loop SOURCES prior to display
		-	Deep Dream
		-	Motion Detector
		-	Overlays
			-	HUD
			-	Alerts
	Processes Game Loop SOURCES into OUTPUTS
		-	compositing
		-	resizing
	Flexibility
		-	SOURCES in the COMPOSER can be any size
			-	SOURCES in the COMPOSER are resized to the COMPOSITE W,H when composited
		-	OUTPUTS from the COMPOSER are resized to the VIEWPORT W,H

------------------------------
What is the VIEWPORT
	An OpenCV window we created to draw on the screen
	An Event Listener for external input
		-	listens for keyboard events every frame
	Flexibility
		-	window can be drawn at any size
		-	supports multiple windows
			-	all windows use the same event listener
		-	provides PREPROCESS inserts
			-	color transform
		-	provides POSTPROCESS inserts
			-	HUD overlay
			-	ALERT overlay



currently,
we almost always write directly to buffer1
-	camera image
	-	fetch camera before entering game loop
	-	return value from deepdream when motion detected during OCTAVECYCLE

-	computed image
	-	initialized with zeros at startup
	-	REM cycle
		-	deepdream return value after non-interrupted REM CYCLE
		-	running inceptionxform (on self) after non-interrupted REM CYCLE
	-	Interrupted REM cycle
			-	composited with buffer2 at beginning of new REM CYCLE if previous cycle was interrupted
	-	OCTAVE CYCLE
		-	neural output from each non-interrupted iteration of OCTAVECYCLE
		-	composited with buffer2 after interrupted OCTAVE CYCLE (is it?)


we sometimes write directly to buffer2
-	initialized with zeros at startup
-	motion was detected during the OCTAVECYCLE
	-	buffer2 stores the last hallucinated frame
	-	deepdream exits and returns camera
	-	buffer1 stores deepdream return value (camera)

2016-11-07 17:03:13
-	Tracking
	-	automate tracking thresholds
	-	trigger events based on "hot" areas of the screen
		-	screen edges?
		-	the cell of an n x n grid with the brightest value?

-	dynamics
	-	cleanup the "programs" currently in use to allow switching between them
	-	cleanup color processing on viewport
	-	add support for automated events

-	display
	-	keep a buffer of final output data (history buffer)
	-	playback history buffer
	-	composite / replace live with history
	-	global frame blending (as in after effects)
		-	 no longer special case new camera image acquisition
		-	 all frames are averaged together before display

-	HUD cleanup
	-	image export
	-	twitter post on demand (?)

AUTOMATE TRACKING THRESHOLDS
every time I set up a session, I have to double check the threshold and lower it or raise it so that motion is detected at a known degree of sensitivity.

if subjects are close to the camera, all the counted values increase

AUTOMATED EVENTS
-	different kind of transforms on REM cycle
	-	scale vertical vs. cale vertical + h, anchor left
	-	value of scale variable
		-	random?
		-	sequenced?
		-	cycle?
		-	related to number of pixels counted during motion detection?
	-	blur multiplier
	-	different feature maps on a given layer

BEST LAYERS
inception_4d_5x5_reduce

2016-11-07 18:06:56
theres a bug with the viewp[ort scaling - its scaling from upper left anchor instead of center anchor

2016-11-07 23:20:57
a more efficient way to resample the viewport?

2016-11-08 01:32:23
added feature map selector to make_step() wow...

2016-11-08 14:45:10
taking a second look at feature map rendering style
priority for this work session is to get tracking automated
better metrics on HUD
fix viewport scaling to work from center instead of top left edge

username
settings
threshold
last
now

2016-11-08 15:40:51
CREATE AN INVENTORY OF EFFECTS. BE ABLE TO SWITCH BETWEEN THEM

for what I'm looking at right now I want to specify:
-	Model
	-	Layer Index
	-	Feature Index
-	Deepdream basic parameters
	-	Iterations
	-	Octaves
	-	Octave cutoff
	-	Octave scale
	-	Step Size
	-	Step Size multiplier
-	Renderer
	-	Make_Step()
	-	Make_Step_Feature_Map()
	-	REM cycle flag (TRUE/FALSE)
		-	TRUE (current value, hallucinate until new camera input)
		-	FALSE (stop processing until new camera input)
-	Modifiers
	-	Feature Index incrementor
	-	Layer Index incrementor
	-	Octave Scale
	-	Pause Motion Detection
	-	REM Transform scale
	-	REM Transform center point

2016-11-08 22:47:32
about to start work on automated tracking. I think that collecting the last n samples of the delta_count then calculating a moving average from that to select the threshold i sthe right place to start

x check the full range of display resolutions available for camera

2. draw stats on screen as needed

3. represent delta count as % is the value the same independent of capture size ?

2016-11-09 02:45:47
I've made the motion detection threshold dynamic - the raw delta count values are placed in fifo queue with fixed length, the queue values are averaged with each step. the threshold value is set to that. Its effective

2016-11-09 03:09:32
working very well, cool feature - still needs a bit of refinement - but playing with some tweaks makes it easier to bias the averaged value. Having them values as history is also cool - the way to deal with closeups is to attenuate large value differences, which should stand out in the sampled delta counts

2016-11-09 10:14:51
if the detector history window width is 2 then its just averaging the last 2 values?

2016-11-09 10:26:52
camera dimensions
# max 2304 x 1536
# 1920 x 1080
# 1600 x 896
# 1280 x 720
# 960 x 720
# 864 x 480
# 800 x 600
# 640 x 480
# 352 x 288
# 320 x 240
# 320 x 180

2016-11-09 11:33:48
maybe there is another way to calculate moton detection?

2016-11-09 12:09:00
maybe. need to move on to other things though. the current method is an improvement

2016-11-09 12:58:45
lookimg so Dark when webcm contrast/exposure is made sorta noir

2016-11-09 15:26:49
Looking at creating "programs" that I can switch between

2016-11-09 16:21:43
how do I "switch" programs now?
Amplifier.set_package('ghost')

what if I could keybind that behavior and switch between?

what is a "program"?
what is the difference between program A and program B?

# 1
A: begins at inception_4d/5x5_reduce
B: begins at inception_5a/5x5_reduce

# 2
A: octave_scale = 1.8
B: octave_scale = 1.2

# 3
A:	uses make_step_featuremap
B:	uses make_step

# 4
A: static mode
B: REM mode

# 5
A: uses 'places' model
B: uses 'cars' model

# 6
A: starts at featuremap[1,2,3,4,5] index 0
B: starts at featuremap[5,10,15] index 2

2016-11-09 20:15:10
moved event listener out of Viewport class is now a global function

2016-11-09 20:24:21
passing the listener function to Viewport instance

2016-11-10 09:36:59
changing layers forces new cycle now, implementing same for featuremaps

2016-11-10 12:03:55
verifying feature map incrementor setup

2016-11-10 12:12:14
works. doing some cleanup before commiting

2016-11-10 15:45:02
modifying the program data definitions

2016-11-10 17:50:12
remaining
x	fix pause functionality
-	work out how to switch programs with program select
x	exporting stills
-	create programs and test

2016-11-10 19:12:55
fixed pause functionality - much less convoluted than previous implement

2016-11-10 21:19:50
assigning layers to those specified in a program's layer list

2016-11-10 23:17:59
pulling layer list from program definitions now

2016-11-10 23:35:53
pulling featuremap list from program definitions now

2016-11-11 08:18:26
Setup 1p at Hotel Zetta. Take the next hour to look at switching network models with progam

2016-11-11 09:32:07
autosaving fully composited frames at end of cycle. still need ability to save on demand

2016-11-11 09:50:37
forced refresh will save the return value from deepdream early exit - this will be a clean camera image. Saving on demand also implemented - this saves whatever is in the image buffer - scaled up to dimensions of viewport.
clean saves of high res material are guaranteed only when a cycle is completed - and that saving is automated

2016-11-11 09:53:05
-	work out how to switch programs with program select
-	create programs and test

2016-11-11 10:50:01
modifying program list data structure

2016-11-11 11:10:21
wtf did that code work the first time through :) :) :)

2016-11-11 11:21:08
adding some error handling to prevent crash when featuremap index goes out of range for a particular layer

2016-11-11 11:24:07
now isnt really the time for that... revisit iof possible

2016-11-11 11:30:46
fixed the problem at featuremap selection

2016-11-11 11:39:46
basic program selection working - need to flesh out further

2016-11-11 12:41:05
model parameters not updating when I switch programs - may need to abandon that idea for now

2016-11-11 12:47:43
simple fix. working now.
give yourself 30 calm minutes to find some cool settings
add more from the location

Turning up the silence in Sci-Fi city at @codame artificial experience show


2017-03-12 15:53:49
rethinking some ideas for the L.A.S.T Festival exhibit


2017-03-14 14:59:45
Its working again, but had to wipe everything. The program was behaving strangely on this new machine. Described above. Despite dreading it, I reinstalled Linux and then the deep learning dev environment. Refined the notes I've been keeping on this. Maybe some new advantages. OpenCV is now at v3.2. I compiled it with option to do linear algebra with Cuda library. It seems faster. but hard to say definitively


2017-03-14 15:43:07
GOALS
CODELOCK
HARDWARE
INSTALLATION
UNINSTALLATION


2017-03-18 17:13:39
I've sourced a mobile AV cart w some accessories and will be placing that order shortly.


2017-03-18 17:36:16
can i make it show only fully rendered frames?
well first of all, lets review how it works



2017-03-19 14:44:13
I have the camera transform working in a test project
probably 
overthinking.
the goal:
rotate the camera input upon reading it

is this a matter of:
1. creating a wrapper function to extend cv2.VideoCapture.read()
2. decorating the existing function?

2017-03-19 15:06:02
motion detection runs in portrait now


2017-03-19 21:23:50
getting close. its a bit of a hack.


2017-03-19 21:33:03
hardcoded portrait orientation is working

NEXT STEPS
- study python function decorators
- cleanup camera orientation functionality
	- choose between portrait and landscape at launch
- examine frame export
	- what's it currently doing?
- investigate the functionality of a Memory class
	- stores rendered output in RAM as an array
	- playback rendered output to framebuffer
- guide images
- switch between 2 cameras
	- camera 1 faces faces the viewer and acts as a mirror
	- camera 2 is behind the screen and acts as a window


2017-03-21 21:34:43
I reimplemented the camera capture system as a threaded queue after a bit of sould searching. Not sure what I was expecting to see. Something? Nothing? Is the system behaving more smoothly now? I can't actually tell. Seems largely the same. How would I be able to tell?
Is there a way for me to profile the systems performance?
- avg. iteration time
- octaves/sec
- current calculation time
- last cycle time


2017-03-21 21:37:58
Is the motion capture not threaded enough? should all of that be happening on a seperate thread? what would be the benefit?
Instead, could it happen on a seperate process?
What if the deepdream computation was happening on a different thread instead, and the output was being queued so that
deepdream.read() would return the next frame in the queue? how woyuld this change the approach to frame buffering?


Python threading doesn't multiprocess - it just makes more efficient use of the CPU
What about the rpyc stuff I'd previously looked at?
The idea is that motion detection and deep dreaming happen in seperate processes
The expected benefit:
computation happens in queues and results fill up a bin
Asynchronous
- decouples display rate from computation rate
display pulls the next available frame from a bin and shows it
may be possible to compute upon bin1.read() + bin2.read()


2017-03-21 23:14:29
looking at rpyc
reading up on python multiprocessing, which offers similar API as Threading

The multiprocessing module comes with plenty of built-in options for building a parallel application. But the three most basic (and safest) are the Process, Queue and Lock classes. 

PROCESS
The Process is an abstraction that sets up another (Python) process, provides it code to run and a way for the parent application to control execution.

from multiprocessing import Process

def say_hello(name='world'):
    print "Hello, %s" % name

p = Process(target=say_hello)
p.start()
p.join()

QUEUES

Queue objects are exactly what they sound like: a thread/process safe, FIFO data structure. They can store any Python object (though simple ones are best) and are extremely useful for sharing data between processes. Queues are especially useful when passed as a parameter to a Process' target function to enable the Process to consume (or return) data.

from multiprocessing import Queue

q = Queue()

q.put('Why hello there!')
q.put(['a', 1, {'b': 'c'}])

q.get() # Returns 'Why hello there!'
q.get() # Returns ['a', 1, {'b': 'c'}]
q.get() # waits for more data to pass through the queue

LOCKS

Like Queue objects, Locks are relatively straightforward. They allow your code to claim the lock, blocking other processes from executing similar code until the process has completed and release the lock. 


2017-03-22 08:29:11
looking at the test application,significant speedups are possible, but how can I apply these ideas to deepdreamvisionquest?
QUEUES:
	next camera frame
	next deepdream render
	next displayed frame

* is it possible to retrieve data from the queue that has already been read? is it random access, or is it sequential by definition?

I'm assuming that queung this information lets me address it asynchronously. For example, show the deepdream rendered que once every second


2017-03-22 08:47:04
who are the workers?

the deepdream function
the capture function

the compose/viewport functio


2017-03-22 19:50:21
I came across some code samples showing a single threaded, multithreaded and multiprocess approach to structuring code. I'm becnhmarking those approaches to see what they do

range: 1,10000,1

SINGLE THREADED
runtime: 50.6829841137s
CPU load during run: 0.96
1 core is in 100% usage throughout

MULTITHREADED 
runtime: 83.9671368599s
2.10
all cores are approx 19-30% in use
* note that this 

MULTIPROCESSOR
runtime: 72.0738749504s
2.05
all cores are approx 30-40% in use
* note that this 

these results are unexpected. taking out the print statements and remeasuring
ending some processes to get a cleaner benchmark

10000, 100000, 100

SINGLE THREADED
runtime: 86.0674118996s

MULTITHREADED (4 threads)
runtime: 89.8049190044s


MULTITHREADED (8 threads)
runtime: 96.2501211166s


MULTITHREADED (16 threads)
runtime: 106.664397955s


MULTIPROCESS (4 processes)
runtime: 88.1409959793s

MULTIPROCESS (12 processes)
runtime: 88.1409959793s


2017-03-22 21:11:22
No room to investigate this much further beyond tonight.My fundamental premise is the right one - decouple computation from display, but implementation is too expensive


2017-03-22 21:23:25
trying one more exercise - slightly different method



2017-03-23 00:47:04
pulled myself away from the optimization experiments and started looking at image processing.
there's a difference between using nd.filters and cv2.filters
getting significantly wider blur fx with some new methods


2017-03-23 00:52:21
ending worksession - frustrating and rewarding at once. there's so much I dont know


2017-03-23 19:01:48
doing some remedial work on numpy before starting this image processing worksession


2017-03-23 21:17:43
wrapped up the remedial work. took a closer look at numpy arrays and matplotlib


2017-03-23 21:18:32
so moving forward - how would I subtract from the output red channel each frame, checking to see if it was 0


2017-03-25 01:05:29
figured out the crash that turned up during last session - or at least figured out how to avoid it. When the number of max iterations is reduced, less time is spent computing. It seems that if the same image is passed to the network at a high enough rate, it crashes. No idea why. Its arcane. Anyway, by scaling the recycled input with a multiplier > 0 the problem doesnt turn up.

More sigfnificantly, I made some exciting progfess towards a more expressive visual language. I costs nothing to color correct and transform the output after its been computed. I'be also tweaked a number of the deepdream make_step parameters. not sure what will stay, still experimenting.

The danger is to get caught in a loop of experimentation though.
1. how can I save the settings and configurations I come across so I can find them again?
2. camera 2
3. what are the controls exposed to participants
4. program USB pad w ASCII sequenceas for controls


2017-03-25 10:28:52
goals (Saturday)
- transforming during the cycle
	- is possibnle? how?
- color correction beta
	-	color
	-	luminance
- image transformation beta
	-	scaling (zooming in)
	-	warping the frame buffer

goals (Sunday)
-	dual cameras
-	guide images


2017-03-25 10:38:44
studying image translation

translation basically means that we are shifting the image by adding/subtracting the X and Y coordinates. In order to do this, we need to create a transformation matrix,

rotation matrix
We can specify the point around which the image would be rotated, the angle of rotation in degrees, and a scaling factor for the image


2017-03-25 18:52:56

reading about 2D convolution (in regard to image processing)

What does "frequency" mean in an image? Well, in this context, frequency refers to the rate of change of pixel values. So we can say that the sharp edges would be high frequency content because the pixel values change rapidly in that region. Going by that logic, plain areas would be low frequency content. Going by this definition, a low pass filter would try to smoothen the edges.


2017-03-25 20:39:01
came across a section on how to make a vignette.
this is probbaly something that should happen at the post rendering stage, similar to the HUD


2017-03-25 21:54:54
came across method for gamma correction


2017-03-25 22:52:24
I've been able to get a kind of color grading happening with what I've learned today. The code is sluggish at the moment as not of what I've added is optimized, and in particular  the vignette that I'm adding is sort of expensive. Also not sure if gamma correction on the input or the output or both is the way to go. I have the computation setup on the camera input, but again, pretty wasteful - is calculating the lookuptable every frame, when it only needs to do so on init and when the gamma value changes.

I'll want to add a control for increasing/decreasing gamma. Interesting to see how it reacts to exposure and brightness set on the Linux camera control utility


2017-03-26 10:08:30
turned off the color grading functionality created yesterday until I can optimize a bit further. 
goals:
	- integrate camera 2
		- realtime switching between cameras
		- composite views
			- overlay
			- 2-up
		- just had an odd idea - is it possible or desirable to create a small snapsot of the camera and composite that on to the viewposrt so that, multiple frames are always shown.
			- further, it would behave liek the queue I'd been thinking about, the most recent capture would always be shown as the last image, and would push older images out of the stack.


2017-03-26 10:12:25
easy enough to access the different cameras by indes. a good start :)
- add a keybind to switch between cameras at runtime


2017-03-26 11:16:41
Just ran it with 2 cameras simultaneously at 1280 x 720. Not wuite sure what the expense was. System seemed responsive - perhaps a bit less fluid than before?


2017-03-26 13:15:01
I'm wanting to replace all the print statements w proper logging

The different levels of logging, from highest urgency to lowest urgency, are:

CRITICAL
ERROR
WARNING
INFO
DEBUG
The setLevel() call sets the minimum log level of messages it actually logs. So if you fh.setLevel(logging.ERROR), then WARNING, INFO, and DEBUG log messages will not be written to the log file (since fh is the log handler for the log file, as opposed to ch which is the handler for the console screen.)

To write a log message in one of these five levels, use the following functions:

logger.critical('This is a critical message.')
logger.error('This is an error message.')
logger.warning('This is a warning message.')
logger.info('This is an informative message.')
logger.debug('This is a low-level debug message.')


2017-03-26 14:00:54
there's a lot more to logging than expected, but I've got a good replacement for all those print statements, now - back to work on that camera!


2017-03-26 14:50:46
realizing that yesterdays tests that I thought were running on a 1920 x 1080 input were actually running on a 1280 x 720 input and scaling up to 1920 (which looked great actually). Those values were hardcoded into the camera instance


2017-03-26 14:52:18
crashes when I set portrait_alignment = False. but why?


2017-03-26 20:23:43
reverted back to the point before I added logging, will continue with print statements as before.


2017-03-26 22:56:58
getting there - able to toggle portrait_alignment without crashing


2017-03-27 11:49:30
struggling with the viewport.
Here's the problem.

I want to be able to toggle portrait_alignment on/off
why?
so that the viewport shapes itself to match whatever the camera is outputting


2017-03-27 17:49:33
making some progress wi the viewport finally


2017-03-27 22:04:44
It seems that the Composer.buffer array needs to be reshaped as well
Hunting this down is so much harder than it needs to be. It points out the fallacy of decentralizing these sources.


2017-03-27 22:11:59
Here's the reality. you need to move on. Forget about the idea of generalizing the camera orientation. Just assume it to be oriented properly when the program starts. It doesn't need to switch back and forth at run time.

With that in mind - I can do something like this:

create camera objects
derive viewport size from camera.width,height
de


2017-03-27 22:37:19

typed in a bunch of stuff maybe it works. 
basic idea was to pull capture_size from the camera object
Viewpoert_size has become an instance of the class Display, which exists at the moment only to house a width and height value accessible thrui dot notation. Once I get it working, I'll replace that bit with a dictionary, or movit back to the Data module.

mocing on, I made global substitutions to reference these changes. At a minimum it shpuld crash in the same way as before.


2017-03-27 23:18:28
its hallucinating on either camera in landscape again
and crashes on portrait just as before - same reasons 


2017-03-27 23:40:22
a bit more detail has it crashing further along. This time it showed the camera output in portrait orientation before crashing. Now that the display size is being adjusted when initialized, the previous conditions I wrote there no lobger applies


2017-03-27 23:43:22
There we go - its working.
I can switch between portraint and landscape modes when the program initializes

Let me try to sum up:

The Display is the onscreen framebuffer.
The display supports 2 aspect ratios:
-	landscape (16:9) default
-	portrait (9:16) transposed counter clockwise
-	the display can be flipped horizontally and vertically in any aspect

The Camera is the framegrabbing and Queing system

Create a camera object, specifying:
- capture_width,
- capture_height
- portrait_alignment 

The Display object relies upon the camera object 
-	display width, height are stored here
-	display width, height are swapped here to match the camera alignment
	-	if the orientation is portrait then the specified display width and height have to be swapped
- the Viewport class relies on knowing the proper Display width, height so:
	- Viewport.show() can scale lower ocataves in buffer1 proportionally to match the display size
- the Composer class relies on knowing the proper Display width and height so:
	- Composer.update() can run inceptionxform on buffer 1
	- buffer2 can resize itself to match the Display width
- show_HUD() relies on knowing the proper Display width, height so:
	-	the function draws HUD updates to a seperate frame buffer that needs to match the DIsplay size

/* I'm noticing that Viewport.show() and Composer.writeBuffer2 basically do the same thing
Viewport.Show() is actually writeBuffer1() + show() */


2017-03-28 00:57:33
cleaned up the logging (print statements) - got rid of most of it
The camera/display issue isnt fully resolved, the frame buffer isnt calling the inceptionxform properly, so it just sits on the same image


2017-03-28 13:28:56
Added expandable logging framework to debug the code better and I guess also as a bit of a programming exercise. Was not happy with my  progress yesterday, enough so that I took a day off from work  to  get my momentum back


2017-03-28 14:56:16
replaced all print statements with logging calls. In a much better position to debug efficiently.


2017-03-28 22:28:52
I have the system up and running on the Monolith. It looks startling. Its amazing. Numerous conclusions to be drawn from the initial test run:
-	the video4Linux Control Panel is only addressing a single camera. Is there a way to run 2 instances of this program and specify the camera?
	- as a fallback, is it possible to emulate aspects of the control panel in software?


2017-03-29 00:34:41
Yes, there is a way to run 2 instances of the video4Linux control panel. There's a lot more functionality there than I knew. It also turns out to be trivial to mirror the captured image vertically and horizontally. Vertical mirroring is surprisingly cool, the forms that emerge are a bit disconnected from your movements.

I'm going to implement camera switching next and then camera flipping
Before getting to that, there's a rendering bug where:
x the sine wave transform doesn't get picked up by the next rendered image, so it appears briefly then disappears instead of propogating
x the octave size frequency mod isn't working
x the inceptionxform funtion isn't calculating the transform correctly


2017-03-29 20:52:59
x	camera switching by pressing F1 / F2
P1 	refine motion detect to allow getting close
P1	sequencing - motion between programs
P1	parameters - what are they - create a list
P2	saving good values - how?
P2	guide images
P3	image capture


2017-03-29 21:18:56
HUD is back online. making the text smaller
am also wondering if possible to output those values to a MatPlotLib window?


2017-03-30 01:19:27
both cameras are running simultaneously. I'm not seeing any real slowdown or latency as a result


2017-03-30 10:24:24
camera switching is working and its cooler than I thought it would be - except for the initial switch, which seems to rely on motyion being detected instead of the viewport being forced to refresh


2017-03-30 11:07:31
camera switching looks like its working - have not fully tested motion detection. looking at that next


2017-03-30 22:26:37
some problems re-addressing camera by index when I switched the USB ports prior to rebooting. Still trying to remember how I brought up the control panels for the cameras

2017-03-31 00:05:32
P1 	refine motion detect to allow getting close
P1	sequencing - motion between programs
P1	parameters - what are they - create a list
P2	saving good values - how?
P2	guide images
P3	image capture
P2	duration based time-out from dreams
	-	forces end of cycle based on how long its been running
P1	transition from end of cycle to beginning of next so the progression is not quite so abrupt


2017-03-31 13:47:00
taking a look at MatplotLib for analyzing what the motion detector is doing .
Keep in mind that there are 2 other areas needing momentum.
P1 	refine motion detect to allow getting close
P1	sequencing - motion between programs
P1	parameters - what are they - create a list



2017-03-31 15:03:21
I found some code to do realtime plotting w matplotlib, looking at it running now. Is pretty CPU intensive. 


2017-03-31 23:18:05
added support for flipping camera input horizontally or vertically


2017-03-31 23:20:49
taking a second look at gamma cprrection


2017-04-01 02:38:11
I put some rough  monitoring in place on teh motion detection viewer, but all it showed me was that the behavior is  more fluid and complicated than I thought. The previous concept of an "overflow" when the current sample and next sample are both over the threshold, doesn't seem to happen any more becaus the threshold is raised by the average value of the last 50 samples

or something like that. In any case its hard to quantify the conditions that result from large motions near the camera. To solve this problem further, I need to look at the data that gets generated.

For tomorrow - put this aside, and work on program settings
- need an explorer setting (which allows me to navigate the space)
- need a way to toggle back and forthe between explorer mode and automated mode
	+  for demonstration
	+  for identifying settings
- need categories for program settings
- need a timer to switch between programs
- need a way to switch network models from within a program
	+ what else needs to be captured within a program?
		* brightness cycle