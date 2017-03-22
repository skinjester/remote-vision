import datetime
from threading import Thread
import cv2

class FPS:
		def __init__(self):
			# store the start time, end time and total number of frames
			# that were examined between the start and end intervals
			self._start = None
			self._end = None
			self._numFrames = 0

		def start(self):
			# start the timer
			self._start = datetime.datetime.now()
			return self

		def stop(self):
			# stop the timer
			self._end = datetime.datetime.now()

		def update(self):
			# increment the total number of frames examined during the
			# start and end intervals
			self._numFrames += 1

		def elapsed(self):
			return (self._end - self._start).total_seconds()

		def fps(self):
			return self._numFrames/self.elapsed()

class WebcamVideoStream:
	def __init__(self, src=0):
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()
		self.stopped = False


	def start(self):
		Thread(target=self.update, args=()).start()
		return self

	def update(self):
		# infinite loop until the thread is stopped
		while True:
			if self.stopped:
				return
			(self.grabbed,self.frame) = self.stream.read()

	def read(self):
		return self.frame


	def stop(self):
		self.stopped = True


