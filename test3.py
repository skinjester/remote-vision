class Car(object):
    wheels = 4

    def __init__(self, make, model):
        self.make = make
        self.model = model

    @staticmethod
    def make_car_sound():
        print 'Vrooom'

mustang = Car('Ford', 'Mustang')
print mustang.wheels
print Car.wheels
print 'hello'
mustang.make_car_sound()
Car.make_car_sound()
