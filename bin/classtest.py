#!/usr/bin/python

class MyStuff(object):
	def __init__(self):
		self.tangerine = "and now a thousand years between"
		self.n = 50
	def apple(self):
		n = self.n
		n+=200
		print 'I am classy apples {}'.format(n)

thing = MyStuff()
print thing.n
thing.apple()
print thing.n
