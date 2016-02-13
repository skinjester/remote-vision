# DeepDream Vision Quest

This script runs the Deep Dream neural visualization on camera input to produce live output. Motion detection is used as a high latency interaction channel. Keyboard input allows dynamic control of neural parameters as the network dreams. My investigation building an engaging experience with this system is the subject of my GDC 2016 Poster Session: [Find Your Spirit Animal In A Deep Dream Vision Quest](http://schedule.gdconf.com/session/find-your-spirit-animal-in-a-deep-dream-vision-quest)

Neural visualization is computationally intensive and the Caffe/OpenCV/CUDA implementation isn't designed for real time output of neural visualization. 30fps output seems out of reach - even at lower resolutions, with reduced iteration rates, running on a fast GPU (TITAN X). From a performance perspective, there would appear to be quite a bit of headroom available. My CPU rarely goes above 20%, and the GPU Load remains under 70%. Is Python itself the bottleneck? Many aspects of this technology are a black box to me, so perhaps further optimizations are possible.

*Design*
The script captures a video frame and will dream about that frame indefinitely, using the output of the last cycle as the input of the next cycle, zooming in slightly each time. The real world soon disappears.

Requirements
-------------


Python
Caffe (and related dependencies)
FFMPEG
CV2 (if you use optical flow)


Reference
-------------
samim23:[DeepDream Animator](https://github.com/samim23/DeepDreamAnim)
graphific: [DeepDream Video](https://github.com/graphific/DeepDreamVideo)
[Diving Deeper into Deep Dreams](http://www.kpkaiser.com/machine-learning/diving-deeper-into-deep-dreams)
[A Visual Introduction to Machine Learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1)

Credits
-------------
Gary Boodhoo | skinjester.com | sciencefictionthriller.com | @skinjester



