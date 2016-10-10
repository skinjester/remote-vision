import mystuff as modularexample


class MyStuff(object):
    def __init__(self):
        self.tangerine = "classy tangerine"

    def apple(self):
        print "I AM CLASSY APPLES"

classexample = MyStuff()
classexample.apple()
print classexample.tangerine


modularexample.apple()
print modularexample.tangerine
