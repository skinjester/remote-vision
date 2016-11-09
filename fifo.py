class Fifo(list):
    def __init__(self):
        self.back = []
        self.append = self.back.append
    def pop(self):
        if not self:
            self.back.reverse()
            self[:] = self.back
            del self.back[:]
        return super(Fifo, self).pop()

if __name__ == '__main__':
    a = Fifo()
    a.append(10)
    a.append(20)
    print a.pop()
    a.append(5)
    print a.pop()
    print a.pop()
    print
# emits: 10 20 5