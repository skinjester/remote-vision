import time

class MyNewClass(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def run(self):
        print '{}:{}'.format(self.a, self.b)
        return

def myfunc(a,b):
    print '{}:{}'.format(a,b)
    return


if __name__ == '__main__':

    # Start consumers
    num_consumers = 10
    print 'Creating {} instances'.format(num_consumers)
    consumers = [ myfunc for i in xrange(num_consumers) ]

    for index,w in enumerate(consumers):
        w(index,index*index)