class MyClass(object):
    def __init__(self):
        self.x = 1

    def DoStuff(self):
        print 'hello', self.x

class InstantiatesMyClass(object):
    def __init__(self):
        self.myClassObj = MyClass()

imc = InstantiatesMyClass()
imc.myClassObj.DoStuff()