import threading


def printit():
    threading.Timer(0.5, printit).start()
    print "Hello, World! {}".format(counter)
    counter += 1
    if counter > 10:
        threading.Timer(0.5, printit).cancel()

counter = 0
printit()