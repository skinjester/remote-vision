# DeepDream Vision Quest
This script runs the Deep Dream neural visualization on camera input to produce live output. Motion detection is used as a high latency interaction channel. Keyboard input allows dynamic control of neural parameters as the network dreams. My investigation building an engaging experience with this system is the subject of my GDC 2016 Poster Session: [Find Your Spirit Animal In A Deep Dream Vision Quest](http://schedule.gdconf.com/session/find-your-spirit-animal-in-a-deep-dream-vision-quest)

Neural visualization is computationally intensive and the Caffe/OpenCV/CUDA implementation isn't designed for real time output of neural visualization. 30fps output seems out of reach - even at lower resolutions, with reduced iteration rates, running on a fast GPU (TITAN X). From a performance perspective, there would appear to be quite a bit of headroom available. My CPU rarely goes above 20%, and the GPU Load remains under 70%. Is Python itself the bottleneck? Many aspects of this technology are a black box to me, so perhaps further optimizations are possible.

## Design
Capture a video frame and dream about that frame indefinitely, using the output of the last cycle as the input of the next cycle, zooming in slightly each time

Detail supplied by the neural network is amplified and aggregated by repeated passes of the source image through the network. With each iteration, neural activation trends toward a local maximum. After each pass, grab another frame from the webcam. Keeping a rolling queue of the last 3 frames captured this way and detect motion by comparing those frames (s)

## Requirements
[What are deep dream images? How do I make my own?](https://www.reddit.com/r/deepdream/comments/3cawxb/what_are_deepdream_images_how_do_i_make_my_own/)

[Running Google’s Deep Dream on Windows – The Easy Way](http://thirdeyesqueegee.com/deepdream/2015/07/19/running-googles-deep-dream-on-windows-with-or-without-cuda-the-easy-way/)

Python

Caffe (and related dependencies)

FFMPEG

CV2 (if you use optical flow)


## References
samim23:[DeepDream Animator](https://github.com/samim23/DeepDreamAnim)
graphific: [DeepDream Video](https://github.com/graphific/DeepDreamVideo)
[Diving Deeper into Deep Dreams](http://www.kpkaiser.com/machine-learning/diving-deeper-into-deep-dreams)
[A Visual Introduction to Machine Learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1)
[Inceptionism: Going Deeper into Neural Networks](http://googleresearch.blogspot.com/2015/06/inceptionism-going-deeper-into-neural.html)

## Credits
Gary Boodhoo | skinjester.com | sciencefictionthriller.com | @skinjester



